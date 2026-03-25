"""HTTP request tools package."""
from seer.tools.base import register_tool
from seer.tools.http.request import HttpRequestTool


def register_http_tools() -> None:
    """Register all HTTP tools."""
    register_tool(HttpRequestTool())


__all__ = ["HttpRequestTool", "register_http_tools"]
