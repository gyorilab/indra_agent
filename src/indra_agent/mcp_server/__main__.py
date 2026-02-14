"""MCP server entry point for INDRA Agent.

Run with: python -m indra_agent.mcp_server

This starts the MCP server using stdio transport, compatible with any MCP client
(Claude Desktop, Claude Code, Cursor, Zed, or custom integrations).

For HTTP deployment, use gunicorn with uvicorn worker (ASGI):
  gunicorn indra_agent.mcp_server.server:app --bind 0.0.0.0:8000 --worker-class uvicorn.workers.UvicornWorker

Or use uvicorn directly:
  uvicorn indra_agent.mcp_server.server:app --host 0.0.0.0 --port 8000

Neo4j credentials are read from environment variables:
  - INDRA_NEO4J_URL
  - INDRA_NEO4J_USER
  - INDRA_NEO4J_PASSWORD
"""
from indra_agent.mcp_server.server import mcp


def main():
    mcp.run()


if __name__ == "__main__":
    main()
