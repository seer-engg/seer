"""
Shared tools package.

Tools are automatically registered when imported.
"""
# Import tools to ensure they're registered
from shared.tools import gmail, google_sheets  # noqa: F401
# Note: model_block removed - use LLM block in workflows instead

# Register GitHub tools (may fail if MCP server not configured)
try:
    from shared.tools.github import register_github_tools
    register_github_tools()
except Exception as e:
    from shared.logger import get_logger
    logger = get_logger("shared.tools")
    logger.warning(f"GitHub tools not available: {e}")

__all__ = ["gmail", "google_sheets"]
