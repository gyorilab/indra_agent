"""MCP server for INDRA Agent.

Exposes biomedical knowledge graph queries through Model Context Protocol.
"""
from indra_agent.mcp_server.server import mcp, main

__all__ = ['mcp', 'main']
