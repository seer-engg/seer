"""
Shared tools package.

Tools are automatically registered when imported.
"""
from seer.tools.airtable import register_airtable_tools  # noqa: F401
from seer.tools.e2b import register_e2b_tools  # noqa: F401
from seer.tools.notion import register_notion_tools  # noqa: F401
from seer.tools.discord import register_discord_tools  # noqa: F401
from seer.tools.github import register_github_tools  # noqa: F401
from seer.tools.http import register_http_tools  # noqa: F401

# Import tools to ensure they're registered
from seer.tools.google import register_google_tools  # noqa: F401
from seer.tools.knowledge import register_knowledge_tools  # noqa: F401
from seer.tools.linkedin import register_linkedin_tools  # noqa: F401
from seer.tools.memory import register_memory_tools  # noqa: F401
from seer.tools.oura import register_oura_tools  # noqa: F401
from seer.tools.postgres import register_postgres_tools  # noqa: F401
from seer.tools.scratch import register_scratch_tools  # noqa: F401
from seer.tools.slack import register_slack_tools  # noqa: F401
from seer.tools.supabase import register_supabase_tools  # noqa: F401
from seer.tools.twilio import register_twilio_tools  # noqa: F401
from seer.tools.websearch import register_websearch_tools  # noqa: F401

# Register tools
register_google_tools()
register_github_tools()
register_postgres_tools()
register_supabase_tools()
register_discord_tools()
register_knowledge_tools()
register_memory_tools()
register_linkedin_tools()
register_oura_tools()
register_slack_tools()
register_airtable_tools()
register_e2b_tools()
register_notion_tools()
register_twilio_tools()
register_websearch_tools()
register_http_tools()
register_scratch_tools()

__all__ = [
    "register_google_tools",
    "register_github_tools",
    "register_postgres_tools",
    "register_supabase_tools",
    "register_discord_tools",
    "register_e2b_tools",
    "register_github_tools",
    "register_google_tools",
    "register_knowledge_tools",
    "register_linkedin_tools",
    "register_memory_tools",
    "register_notion_tools",
    "register_oura_tools",
    "register_slack_tools",
    "register_supabase_tools",
    "register_twilio_tools",
    "register_websearch_tools",
    "register_http_tools",
    "register_scratch_tools",
]
