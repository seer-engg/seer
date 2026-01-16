"""Integration authentication tools for MCP server."""

import webbrowser
from typing import List

from mcp.types import TextContent, Tool  # pylint: disable=no-name-in-module # Reason: external MCP SDK

from api.integrations.services import list_connections
from shared.config import config
from shared.database import User


async def seer_list_integrations(
    user: User,
) -> List[TextContent]:
    """List available integrations and their connection status.

    Args:
        user: Authenticated user

    Returns:
        List of text content with integration status
    """
    connections = await list_connections(user)

    if not connections:
        return [
            TextContent(
                type="text",
                text="""No integrations connected yet.

Use seer_configure_integration_auth to connect integrations like GitHub, Slack, Google, etc.""",
            )
        ]

    integration_info = []
    for conn in connections:
        status = "✅ Connected" if conn.status == "active" else "❌ Disconnected"
        scopes = conn.scopes[:100] if conn.scopes else "No scopes"

        integration_info.append(f"""Provider: {conn.provider}
Status: {status}
Account: {conn.provider_account_id or "Unknown"}
Scopes: {scopes}""")

    integrations_text = "\n\n".join(integration_info)

    return [
        TextContent(
            type="text",
            text=f"""Your connected integrations:

{integrations_text}

Use seer_configure_integration_auth to add more integrations.""",
        )
    ]


async def seer_configure_integration_auth(
    integration: str,
) -> List[TextContent]:
    """Get OAuth URL for integration setup.

    This tool helps you connect integrations (like GitHub, Slack, Google)
    to use in your workflows. It will open your browser to complete the
    OAuth authorization process.

    Args:
        integration: Integration provider (e.g., "github", "slack", "google")

    Returns:
        List of text content with OAuth URL
    """
    # Map integration names to OAuth providers
    provider_map = {
        "github": "github",
        "slack": "slack",
        "google": "google",
        "linear": "linear",
        "notion": "notion",
        "supabase": "supabase",
    }

    provider = provider_map.get(integration.lower())
    if not provider:
        available_integrations = ", ".join(provider_map.keys())
        return [
            TextContent(
                type="text",
                text=f"""Unknown integration: {integration}

Available integrations: {available_integrations}""",
            )
        ]

    # Default scopes for common providers
    scope_map = {
        "github": "repo,read:user,user:email",
        "slack": "channels:read,chat:write,users:read",
        "google": "https://www.googleapis.com/auth/gmail.readonly https://www.googleapis.com/auth/drive.readonly",
        "linear": "read,write",
        "notion": "read,update,insert",
        "supabase": "all",
    }

    scope = scope_map.get(provider, "")

    # Construct OAuth authorization URL
    frontend_url = config.FRONTEND_URL or "http://localhost:3000"
    redirect_to = f"{frontend_url}/settings/integrations"

    auth_url = f"{config.API_URL}/integrations/{provider}/authorize?scope={scope}&redirect_to={redirect_to}"

    # Try to open browser automatically
    try:
        webbrowser.open(auth_url)
        opened_browser = True
    except Exception:  # pylint: disable=broad-exception-caught # Reason: Graceful degradation if browser unavailable
        opened_browser = False

    browser_msg = "\n\n✅ Opening your browser automatically..." if opened_browser else "\n\n⚠️ Please copy and paste the URL into your browser."

    return [
        TextContent(
            type="text",
            text=f"""To connect {integration.title()}, please authorize at:

{auth_url}{browser_msg}

After authorizing, you'll be redirected to the Seer dashboard.
The integration will then be available for use in your workflows.

Use seer_list_integrations to verify the connection was successful.""",
        )
    ]


def get_integration_tools() -> List[Tool]:
    """Get integration auth tool definitions for MCP."""
    return [
        Tool(
            name="seer_list_integrations",
            description="List all connected integrations and their status",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="seer_configure_integration_auth",
            description="Get OAuth authorization URL to connect an integration (GitHub, Slack, Google, etc.)",
            inputSchema={
                "type": "object",
                "properties": {
                    "integration": {
                        "type": "string",
                        "description": "Integration provider name (github, slack, google, linear, notion, supabase)",
                        "enum": ["github", "slack", "google", "linear", "notion", "supabase"],
                    },
                },
                "required": ["integration"],
            },
        ),
    ]
