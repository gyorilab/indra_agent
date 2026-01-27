"""MCP server for INDRA Agent.

Exposes biomedical knowledge graph queries through Model Context Protocol.
"""
from indra_agent.mcp_server.server import mcp, app, get_client

__all__ = ['mcp', 'app', 'get_client']
