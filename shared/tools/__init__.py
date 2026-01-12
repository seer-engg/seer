"""
Shared tools package.

Tools are automatically registered when imported.
"""
from shared.config import config
from shared.tools.github import register_github_tools  # noqa: F401

# Import tools to ensure they're registered
from shared.tools.google import register_google_tools  # noqa: F401
from shared.tools.reasoning import register_reasoning_tools  # noqa: F401
from shared.tools.supabase import register_supabase_tools  # noqa: F401

# Register tools
register_google_tools()
register_github_tools()
register_supabase_tools()

# Conditionally register RLM tools based on feature flag
if config.enable_rlm_tool:
    register_reasoning_tools()

__all__ = [
    "register_google_tools",
    "register_github_tools",
    "register_supabase_tools",
    "register_reasoning_tools",
]
