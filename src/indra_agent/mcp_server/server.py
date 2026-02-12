"""MCP server with progressive disclosure pattern for knowledge graph navigation.

Architecture:
- Layer 1: Schema Discovery (get_graph_schema)
- Layer 2: Query Execution (execute_cypher)
- Layer 3: Validation (validate_cypher)
- Layer 4: Enrichment (enrich_results)
- Layer 5: Gateway Tools (4 tools for graph navigation):
    - ground_entity: Natural language → CURIEs via GILDA
    - suggest_endpoints: Graph navigation suggestions
    - call_endpoint: Execute any autoclient function
    - get_navigation_schema: Full edge map for discovery
"""
import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.templating import Jinja2Templates

from indra_cogex.client.neo4j_client import Neo4jClient
from indra.config import get_config
from indra_agent.mcp_server.enrichment import enrich_results, DisclosureLevel
from indra_agent.mcp_server.schema_discovery import get_graph_schema
from indra_agent.mcp_server.query_execution import execute_cypher
from indra_agent.mcp_server.validation import validate_cypher
from indra_agent.mcp_server.cache import cache as _cache, make_key


def compact_json(obj: Any) -> str:
    """Serialize to compact JSON for token efficiency."""
    return json.dumps(obj, separators=(',', ':'), default=str)

logger = logging.getLogger(__name__)

# Read allowed hosts and origins from environment variables
_allowed_hosts_env = get_config("MCP_ALLOWED_HOSTS", failure_ok=False)
_allowed_origins_env = get_config("MCP_ALLOWED_ORIGINS", failure_ok=False)

# Parse comma-separated values into lists
_allowed_hosts = [h.strip() for h in _allowed_hosts_env.split(",") if h.strip()]
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

# Parse stateless mode from environment (default: True for production)
_stateless = os.getenv("MCP_STATELESS", "true").lower() in ("true", "t", "1", "yes", "on")

# Parse JSON response mode from environment (default: True)
_json_response = os.getenv("MCP_JSON_RESPONSE", "true").lower() in ("true", "t", "1", "yes", "on")

# Initialize MCP server with stateless mode for horizontal scaling
mcp = FastMCP(
    "indra_cogex",
    stateless_http=_stateless,
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_allowed_hosts,
        allowed_origins=_allowed_origins,
    )
)

# Initialize Jinja2 templates
templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "templates")
)

# Neo4j client singleton
_neo4j_client = None
_client_lock = threading.RLock()


def get_client() -> Neo4jClient:
    """Get or create singleton Neo4j client (thread-safe)."""
    global _neo4j_client
    with _client_lock:
        if _neo4j_client is None:
            _neo4j_client = Neo4jClient(
                get_config("INDRA_NEO4J_URL"),
                auth=(get_config("INDRA_NEO4J_USER"), get_config("INDRA_NEO4J_PASSWORD"))
            )
            logger.info(f"Initialized Neo4j client: {get_config('INDRA_NEO4J_URL')}")
        return _neo4j_client


# ============================================================================
# Custom Routes
# ============================================================================

@mcp.custom_route("/", methods=["GET"])
async def landing_page(request):
    """Serve HTML landing page at root path."""
    return templates.TemplateResponse(
        "landing_page.html",
        {"request": request, "server_name": mcp.name}
    )


@mcp.custom_route("/mcp", methods=["GET"])
async def mcp_landing_page(request):
    """Handle GET requests to /mcp endpoint.

    When deployed behind CloudFront at mydomain.com/mcp, this handler:
    - Serves HTML landing page for browser GET requests (Accept: text/html)
    - Returns JSON info/health response for API client GET requests
    - POST requests to /mcp are handled by FastMCP's MCP protocol endpoint
    """
    accept_header = request.headers.get("accept", "").lower()
    is_browser_request = "text/html" in accept_header or not accept_header

    if is_browser_request:
        # Serve landing page for browser GET requests
        return templates.TemplateResponse(
            "landing_page.html",
            {"request": request, "server_name": mcp.name}
        )
    else:
        # For API clients, health checks, or other non-browser GET requests
        # Return a helpful JSON response
        return JSONResponse({
            "service": "indra_cogex",
            "status": "healthy",
            "endpoint": "/mcp",
            "protocol": "MCP (Model Context Protocol)",
            "transport": "HTTP POST",
            "message": "This endpoint accepts MCP protocol messages via POST requests. "
                      "For HTML documentation, send a GET request with Accept: text/html header."
        })


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """Health check endpoint for monitoring."""
    return JSONResponse({"status": "healthy", "service": "indra_cogex"})


# ============================================================================
# Input Models
# ============================================================================

class GetGraphSchemaInput(BaseModel):
    """Input model for get_graph_schema."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    detail_level: str = Field(
        "summary",
        description="Schema detail level: summary, entity_types, relationship_types, patterns, full"
    )
    entity_type: Optional[str] = Field(
        None,
        description="Optional: Filter to specific entity type (e.g., 'Gene', 'Disease')"
    )
    relationship_type: Optional[str] = Field(
        None,
        description="Optional: Filter to specific relationship type"
    )


class ExecuteCypherInput(BaseModel):
    """Input model for execute_cypher."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    query: str = Field(..., description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameterized query values (prevents injection)"
    )
    validate_first: bool = Field(True, description="Run validation before execution")
    timeout_ms: int = Field(30000, ge=1000, le=120000, description="Query timeout in milliseconds")
    max_results: int = Field(100, ge=1, le=10000, description="Maximum result rows")
    explain: bool = Field(False, description="Return query plan instead of executing")
    offset: int = Field(0, ge=0, description="Starting offset for pagination (default: 0)")
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Maximum items per response page. Auto-truncates to ~20k tokens if exceeded."
    )


class ValidateCypherInput(BaseModel):
    """Input model for validate_cypher."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    query: str = Field(..., description="Cypher query to validate")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")


class EnrichResultsInput(BaseModel):
    """Input model for enrich_results."""
    model_config = ConfigDict(validate_assignment=True, extra='forbid')

    results: List[Any] = Field(..., description="Query results to enrich")
    disclosure_level: str = Field(
        "standard",
        description="Metadata level: minimal, standard, detailed, exploratory"
    )
    result_type: Optional[str] = Field(
        None,
        description="Type of results (e.g., 'gene', 'pathway')"
    )
    offset: int = Field(0, ge=0, description="Starting offset for pagination (default: 0)")
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=10000,
        description="Maximum items per response page. Auto-truncates to ~20k tokens if exceeded."
    )


# ============================================================================
# Layer 1: Schema Discovery
# ============================================================================

@mcp.tool(
    name="get_graph_schema",
    annotations={"title": "Get Graph Schema", "readOnlyHint": True, "idempotentHint": True}
)
async def get_graph_schema_tool(params: GetGraphSchemaInput) -> str:
    """Progressively discover Neo4j graph schema (entity types, relationships).

    Returns
    -------
    :
        JSON with schema metadata at requested detail level
    """
    _SCHEMA_TTL = 86400  # 24 hours — schema changes are rare

    cache_key = make_key("schema", params.detail_level, params.entity_type, params.relationship_type)

    # Check cache first
    try:
        cached = await asyncio.to_thread(_cache.get, cache_key)
    except Exception:
        cached = None

    if cached is not None:
        logger.debug("Schema cache hit for %s", cache_key)
        return compact_json(cached)

    try:
        # Add 60 second timeout to prevent hangs on expensive queries
        result = await asyncio.wait_for(
            asyncio.to_thread(
                get_graph_schema,
                client=get_client(),
                detail_level=params.detail_level,
                entity_type=params.entity_type,
                relationship_type=params.relationship_type
            ),
            timeout=60.0
        )

        # Cache the result with long TTL
        try:
            await asyncio.to_thread(
                lambda: _cache.set(cache_key, result, expire=_SCHEMA_TTL, tag="schema")
            )
        except Exception:
            logger.debug("Schema cache write failed for %s", cache_key)

        return compact_json(result)
    except asyncio.TimeoutError:
        logger.error("Schema discovery timed out after 60 seconds")
        return compact_json({
            "error": "Schema discovery timed out after 60 seconds",
            "hint": "Try a simpler detail_level like 'summary' or filter by entity_type/relationship_type"
        })
    except Exception as e:
        logger.error(f"Schema discovery failed: {e}", exc_info=True)
        return compact_json({"error": f"Schema discovery failed: {str(e)}"})


# ============================================================================
# Layer 2: Query Execution
# ============================================================================

@mcp.tool(
    name="execute_cypher",
    annotations={"title": "Execute Cypher Query", "readOnlyHint": True, "idempotentHint": True}
)
async def execute_cypher_tool(params: ExecuteCypherInput) -> str:
    """Execute arbitrary Cypher query on CoGEx knowledge graph.

    Returns
    -------
    :
        JSON with query results or error information
    """
    # NOTE: BioEntity IDs in Neo4j use lowercase namespace prefixes
    # (e.g., "hgnc:11179" not "HGNC:11179"). Use norm_id() conventions.
    try:
        # Validate query if requested
        if params.validate_first:
            validation = await asyncio.to_thread(
                validate_cypher,
                query=params.query,
                parameters=params.parameters
            )
            if not validation.get("valid", False):
                return compact_json({
                    "error": "Query validation failed",
                    "validation": validation
                })

        # Execute query with pagination
        result = await asyncio.to_thread(
            execute_cypher,
            client=get_client(),
            query=params.query,
            parameters=params.parameters,
            timeout_ms=params.timeout_ms,
            max_results=params.max_results,
            explain=params.explain,
            offset=params.offset,
            limit=params.limit,
        )
        return compact_json(result)

    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        return compact_json({"error": f"Query execution failed: {str(e)}"})


# ============================================================================
# Layer 3: Validation
# ============================================================================

@mcp.tool(
    name="validate_cypher",
    annotations={"title": "Validate Cypher Query", "readOnlyHint": True, "idempotentHint": True}
)
async def validate_cypher_tool(params: ValidateCypherInput) -> str:
    """Validate Cypher query safety before execution.

    Returns
    -------
    :
        JSON with validation results
    """
    try:
        result = await asyncio.to_thread(
            validate_cypher,
            query=params.query,
            parameters=params.parameters
        )
        return compact_json(result)
    except Exception as e:
        logger.error(f"Query validation failed: {e}", exc_info=True)
        return compact_json({"error": f"Validation failed: {str(e)}"})


# ============================================================================
# Layer 4: Result Enrichment
# ============================================================================

@mcp.tool(
    name="enrich_results",
    annotations={"title": "Enrich Query Results", "readOnlyHint": True, "idempotentHint": True}
)
async def enrich_results_tool(params: EnrichResultsInput) -> str:
    """Add progressive metadata to query results.

    Returns
    -------
    :
        JSON with enriched results
    """
    try:
        try:
            disclosure_level = DisclosureLevel(params.disclosure_level)
        except ValueError:
            return compact_json({
                "error": f"Invalid disclosure_level '{params.disclosure_level}'",
                "valid_levels": ["minimal", "standard", "detailed", "exploratory"]
            })

        result = await asyncio.to_thread(
            enrich_results,
            results=params.results,
            disclosure_level=disclosure_level,
            result_type=params.result_type,
            client=get_client(),
            offset=params.offset,
            limit=params.limit,
        )
        return compact_json(result)

    except Exception as e:
        logger.error(f"Result enrichment failed: {e}", exc_info=True)
        return compact_json({"error": f"Enrichment failed: {str(e)}"})


# ============================================================================
# Layer 5: Gateway Tools for Graph Navigation
# ============================================================================

def _register_gateway_tools():
    """Register gateway tools for graph navigation.

    Exposes 4 tools instead of 100+ individual functions:
    1. ground_entity - Natural language → CURIEs
    2. suggest_endpoints - Graph navigation suggestions
    3. call_endpoint - Execute any autoclient function
    4. get_navigation_schema - Full edge map for discovery
    """
    try:
        from indra_agent.mcp_server.autoclient_tools import register_gateway_tools
        count = register_gateway_tools(mcp, get_client)
        logger.info(f"Registered {count} gateway tools for graph navigation")
    except Exception as e:
        logger.warning(f"Failed to register gateway tools: {e}")


# Register gateway tools when module is imported
_register_gateway_tools()


# ============================================================================
# ASGI App for Gunicorn/Uvicorn
# ============================================================================
# Note: This is an ASGI application. When using gunicorn, you must use
# the uvicorn worker class:
#   gunicorn indra_agent.mcp_server.server:app --worker-class uvicorn.workers.UvicornWorker
#
# Or use uvicorn directly:
#   uvicorn indra_agent.mcp_server.server:app --host 0.0.0.0 --port 8000
# ============================================================================

# Configure JSON response mode
mcp.settings.json_response = _json_response

# Create base ASGI app using FastMCP's built-in streamable_http_app() method
# This is the recommended approach per FastMCP documentation
_base_app = mcp.streamable_http_app()


# ============================================================================
# Middleware to handle GET requests to /mcp endpoint
# ============================================================================

class MCPLandingPageMiddleware(BaseHTTPMiddleware):
    """Middleware to intercept GET requests to /mcp and serve landing page.

    This ensures GET requests to /mcp are handled before FastMCP's route,
    allowing us to serve the landing page for browsers while letting POST
    requests pass through to FastMCP's MCP protocol handler.
    """

    async def dispatch(self, request: Request, call_next):
        # Intercept GET requests to /mcp
        if request.method == "GET" and request.url.path == "/mcp":
            accept_header = request.headers.get("accept", "").lower()
            is_browser_request = "text/html" in accept_header or not accept_header

            if is_browser_request:
                # Serve landing page for browser GET requests
                return templates.TemplateResponse(
                    "landing_page.html",
                    {"request": request, "server_name": mcp.name}
                )
            else:
                # For API clients, health checks, or other non-browser GET requests
                # Return a helpful JSON response
                return JSONResponse({
                    "service": "indra_cogex",
                    "status": "healthy",
                    "endpoint": "/mcp",
                    "protocol": "MCP (Model Context Protocol)",
                    "transport": "HTTP POST",
                    "message": "This endpoint accepts MCP protocol messages via POST requests. "
                              "For HTML documentation, send a GET request with Accept: text/html header."
                })

        # For all other requests, pass through to the underlying app
        return await call_next(request)


# Wrap the base app with middleware to handle /mcp GET requests
app = MCPLandingPageMiddleware(_base_app)

# Ensure diskcache is flushed on worker shutdown (gunicorn SIGTERM, etc.)
import atexit
atexit.register(lambda: _cache.close()  # type: ignore[union-attr]
                if _cache is not None else None)

__all__ = ['mcp', 'get_client', 'app']
