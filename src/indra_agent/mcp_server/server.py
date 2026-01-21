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
import contextlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import click
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Mount, Route

from indra_cogex.client.neo4j_client import Neo4jClient
from indra.config import get_config
from indra_agent.mcp_server.enrichment import enrich_results, DisclosureLevel
from indra_agent.mcp_server.schema_discovery import get_graph_schema
from indra_agent.mcp_server.query_execution import execute_cypher
from indra_agent.mcp_server.validation import validate_cypher

logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("indra_cogex")

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
# Landing Page HTML
# ============================================================================

def get_landing_page_html(server_name: str = "indra_cogex") -> str:
    """Generate HTML landing page for the MCP server."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{server_name} - MCP Server</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .container {{
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }}
        h1 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 30px;
            font-size: 1.2em;
        }}
        .info-section {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info-section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        .info-section p {{
            margin-bottom: 10px;
        }}
        .endpoint {{
            background: #e9ecef;
            padding: 10px 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            word-break: break-all;
        }}
        .status {{
            display: inline-block;
            padding: 5px 15px;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        ul {{
            margin-left: 20px;
            margin-top: 10px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{server_name}</h1>
        <p class="subtitle">Model Context Protocol (MCP) Server</p>
        
        <div class="status">✓ Server is running</div>
        
        <div class="info-section">
            <h2>About</h2>
            <p>This is an MCP server providing access to the INDRA CoGEx knowledge graph. 
            The server exposes tools, resources, and prompts for querying and navigating 
            biomedical knowledge graphs.</p>
        </div>
        
        <div class="info-section">
            <h2>Endpoint</h2>
            <p>MCP clients should connect to this endpoint:</p>
            <div class="endpoint">POST /mcp</div>
            <p>This endpoint accepts MCP protocol messages via POST requests.</p>
        </div>
        
        <div class="info-section">
            <h2>Features</h2>
            <ul>
                <li><strong>Schema Discovery:</strong> Explore graph structure and entity types</li>
                <li><strong>Query Execution:</strong> Execute Cypher queries on the knowledge graph</li>
                <li><strong>Query Validation:</strong> Validate queries before execution</li>
                <li><strong>Result Enrichment:</strong> Add metadata to query results</li>
                <li><strong>Graph Navigation:</strong> Navigate the knowledge graph using gateway tools</li>
            </ul>
        </div>
        
        <div class="info-section">
            <h2>Transport</h2>
            <p>This server uses the <strong>Streamable HTTP</strong> transport mode, 
            which supports both stateless and stateful operation.</p>
        </div>
        
        <div class="footer">
            <p>INDRA CoGEx MCP Server | Model Context Protocol</p>
        </div>
    </div>
</body>
</html>"""


async def landing_page_handler(request):
    """Handle GET requests to /mcp with HTML landing page."""
    return HTMLResponse(content=get_landing_page_html(mcp.name))


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
        return json.dumps(result, indent=2)
    except asyncio.TimeoutError:
        logger.error("Schema discovery timed out after 60 seconds")
        return json.dumps({
            "error": "Schema discovery timed out after 60 seconds",
            "hint": "Try a simpler detail_level like 'summary' or filter by entity_type/relationship_type"
        }, indent=2)
    except Exception as e:
        logger.error(f"Schema discovery failed: {e}", exc_info=True)
        return json.dumps({"error": f"Schema discovery failed: {str(e)}"}, indent=2)


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
    try:
        # Validate query if requested
        if params.validate_first:
            validation = await asyncio.to_thread(
                validate_cypher,
                query=params.query,
                parameters=params.parameters
            )
            if not validation.get("valid", False):
                return json.dumps({
                    "error": "Query validation failed",
                    "validation": validation
                }, indent=2)

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
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=True)
        return json.dumps({"error": f"Query execution failed: {str(e)}"}, indent=2)


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
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Query validation failed: {e}", exc_info=True)
        return json.dumps({"error": f"Validation failed: {str(e)}"}, indent=2)


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
            return json.dumps({
                "error": f"Invalid disclosure_level '{params.disclosure_level}'",
                "valid_levels": ["minimal", "standard", "detailed", "exploratory"]
            }, indent=2)

        result = await asyncio.to_thread(
            enrich_results,
            results=params.results,
            disclosure_level=disclosure_level,
            result_type=params.result_type,
            client=get_client(),
            offset=params.offset,
            limit=params.limit,
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Result enrichment failed: {e}", exc_info=True)
        return json.dumps({"error": f"Enrichment failed: {str(e)}"}, indent=2)


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


@click.command()
@click.option(
    "--http",
    is_flag=True,
    default=False,
    help="Run server in HTTP streamable transport mode (for network access). "
         "Default is stdio transport mode (for local clients)."
)
@click.option(
    "--host",
    default=None,
    help="Host to bind to when using HTTP transport. "
         "Defaults to 0.0.0.0 (all interfaces) or from MCP_HOST environment variable."
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to bind to when using HTTP transport. "
         "Defaults to 8000 or from MCP_PORT environment variable."
)
@click.option(
    "--stateless/--stateful",
    default=True,
    help="Use stateless HTTP mode (default: True). "
         "Stateless mode is recommended for easier scaling. "
         "Use --stateful to enable stateful mode."
)
@click.option(
    "--json-response/--streaming",
    default=True,
    help="Use JSON response mode instead of streaming (default: True). "
         "Simpler for most use cases. Use --streaming to enable SSE streaming."
)
def main(http: bool, host: Optional[str], port: Optional[int], stateless: bool, json_response: bool):
    """INDRA CoGEx MCP Server.

    Run the MCP server in either stdio mode (default, for local clients) or
    HTTP streamable transport mode (for network access).

    Examples:

    \b
    # Run in stdio mode (default, for Claude Desktop, Cursor, etc.)
    indra-agent

    \b
    # Run in HTTP mode for network access
    indra-agent --http

    \b
    # Run in HTTP mode with custom host/port
    indra-agent --http --host 0.0.0.0 --port 8000
    """
    if http:
        # HTTP streamable transport mode for network access
        http_host = host or os.getenv("MCP_HOST", "0.0.0.0")
        http_port = port or int(os.getenv("MCP_PORT", "8000"))

        logger.info(
            f"Starting MCP server in HTTP streamable transport mode on {http_host}:{http_port}"
        )
        logger.info(f"Stateless mode: {stateless}, JSON response: {json_response}")

        # Update FastMCP settings before running
        mcp.settings.host = http_host
        mcp.settings.port = http_port
        mcp.settings.stateless_http = stateless
        mcp.settings.json_response = json_response

        # Create Starlette app with landing page and MCP server
        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette):
            """Manage MCP session manager lifecycle."""
            async with mcp.session_manager.run():
                yield

        # Get the MCP server's ASGI app (handles routes at /mcp by default)
        # Note: stateless_http and json_response are already set via mcp.settings above
        mcp_app = mcp.streamable_http_app()

        # Create Starlette app for lifespan management
        starlette_app = Starlette(routes=[], lifespan=lifespan)
        
        # Create ASGI wrapper that intercepts GET /mcp and forwards rest to MCP app
        class CombinedASGIApp:
            """ASGI app that serves landing page for GET /mcp, forwards rest to MCP app."""
            
            def __init__(self, mcp_app, landing_handler, starlette_app):
                self.mcp_app = mcp_app
                self.landing_handler = landing_handler
                self.starlette_app = starlette_app
            
            async def __call__(self, scope, receive, send):
                if scope["type"] == "lifespan":
                    # Use Starlette's lifespan handling
                    await self.starlette_app(scope, receive, send)
                elif scope["type"] == "http":
                    path = scope.get("path", "")
                    method = scope.get("method", "")
                    
                    # Intercept GET /mcp for landing page
                    if path == "/mcp" and method == "GET":
                        from starlette.requests import Request
                        request = Request(scope, receive)
                        response = await self.landing_handler(request)
                        await response(scope, receive, send)
                    else:
                        # Forward all other requests to MCP app
                        await self.mcp_app(scope, receive, send)
                else:
                    await self.mcp_app(scope, receive, send)
        
        app = CombinedASGIApp(mcp_app, landing_page_handler, starlette_app)

        # Run with uvicorn
        try:
            import uvicorn
            uvicorn.run(app, host=http_host, port=http_port, log_level="info")
        except ImportError:
            logger.error("uvicorn is required for HTTP mode. Install it with: pip install uvicorn")
            raise
    else:
        # Default stdio transport mode for local clients
        logger.info("Starting MCP server in stdio transport mode")
        mcp.run()


__all__ = ['mcp', 'get_client', 'main']
