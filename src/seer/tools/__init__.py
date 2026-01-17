"""
Shared tools package.

Tools are automatically registered when imported.
"""
from seer.tools.github import register_github_tools  # noqa: F401

# Import tools to ensure they're registered
from seer.tools.google import register_google_tools  # noqa: F401
from seer.tools.supabase import register_supabase_tools  # noqa: F401

# Register tools
register_google_tools()
register_github_tools()
register_supabase_tools()

__all__ = [
    "register_google_tools",
    "register_github_tools",
    "register_supabase_tools",
]
