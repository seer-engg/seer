"""
Shared tools package.

Tools are automatically registered when imported.
"""
# Import tools to ensure they're registered
from shared.tools.google import register_google_tools  # noqa: F401
from shared.tools.github import register_github_tools  # noqa: F401




# Register tools
register_google_tools()
register_github_tools()

__all__ = [
    "register_google_tools",
    "register_github_tools",
]
