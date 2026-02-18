"""
Integration metadata service.

Provides centralized configuration and aggregation of integration metadata
for the frontend to consume dynamically.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from seer.logger import get_logger
from seer.tools.base import list_tools

logger = get_logger("services.integrations.metadata")


# ==============================================================================
# INTEGRATION CONFIGURATIONS
# ==============================================================================
# Static configuration for all supported integrations.
# When adding a new integration, add its configuration here.

INTEGRATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Google integrations
    "gmail": {
        "display_name": "Gmail",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/gmail.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#EA4335",
        "default_scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/gmail.readonly",
                "display_name": "Read emails",
                "description": "Read email messages and labels"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.send",
                "display_name": "Send emails",
                "description": "Send email messages on your behalf"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.compose",
                "display_name": "Compose drafts",
                "description": "Create and modify email drafts"
            },
            {
                "value": "https://www.googleapis.com/auth/gmail.modify",
                "display_name": "Modify emails",
                "description": "Modify email labels and read/unread status"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gmail_"],
            "scope_keywords": ["gmail"]
        }
    },
    "google_drive": {
        "display_name": "Google Drive",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/drive.google.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#4285F4",
        "default_scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/drive.readonly",
                "display_name": "Read files",
                "description": "View files and folders in Google Drive"
            },
            {
                "value": "https://www.googleapis.com/auth/drive.file",
                "display_name": "Manage files",
                "description": "Create, edit, and delete files created by this app"
            },
            {
                "value": "https://www.googleapis.com/auth/drive",
                "display_name": "Full access",
                "description": "Full read/write access to all Google Drive files"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gdrive_", "google_drive_"],
            "scope_keywords": ["drive"]
        }
    },
    "google_sheets": {
        "display_name": "Google Sheets",
        "oauth_provider": "google",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/sheets.google.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#0F9D58",
        "default_scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
        "scopes": [
            {
                "value": "https://www.googleapis.com/auth/spreadsheets.readonly",
                "display_name": "Read spreadsheets",
                "description": "View spreadsheets and their data"
            },
            {
                "value": "https://www.googleapis.com/auth/spreadsheets",
                "display_name": "Edit spreadsheets",
                "description": "Read and write spreadsheet data"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["gsheets_", "google_sheets_"],
            "scope_keywords": ["spreadsheets"]
        }
    },
    # GitHub integrations
    "github": {
        "display_name": "GitHub",
        "oauth_provider": "github",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/github.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#24292F",
        "default_scopes": ["repo"],
        "scopes": [
            {
                "value": "repo",
                "display_name": "Repository access",
                "description": "Full control of private repositories"
            },
            {
                "value": "user:email",
                "display_name": "Email access",
                "description": "Access email addresses (read-only)"
            },
            {
                "value": "read:user",
                "display_name": "Profile access",
                "description": "Read user profile data"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["github_"],
            "scope_keywords": ["repo", "github"]
        }
    },
    "pull_request": {
        "display_name": "GitHub Pull Requests",
        "oauth_provider": "github",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/github.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#24292F",
        "default_scopes": ["repo"],
        "scopes": [
            {
                "value": "repo",
                "display_name": "Repository access",
                "description": "Full control of private repositories"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["github_pull_request", "github_list_pull"],
            "scope_keywords": ["repo"]
        }
    },
    # LinkedIn integration
    "linkedin": {
        "display_name": "LinkedIn",
        "oauth_provider": "linkedin",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/linkedin.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#0A66C2",
        "default_scopes": ["openid", "profile", "email"],
        "scopes": [
            {
                "value": "openid",
                "display_name": "OpenID",
                "description": "Authenticate with OpenID Connect"
            },
            {
                "value": "profile",
                "display_name": "Profile",
                "description": "Access your LinkedIn profile"
            },
            {
                "value": "email",
                "display_name": "Email",
                "description": "Access your email address"
            },
            {
                "value": "w_member_social",
                "display_name": "Post content",
                "description": "Create and share posts on your behalf"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["linkedin_"],
            "scope_keywords": ["w_member_social", "linkedin"]
        }
    },
    # Discord integration
    "discord": {
        "display_name": "Discord",
        "oauth_provider": "discord",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/discord.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#5865F2",
        "default_scopes": ["bot"],
        "scopes": [
            {
                "value": "bot",
                "display_name": "Bot access",
                "description": "Install bot to your Discord server"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["discord_"],
            "scope_keywords": ["discord"]
        }
    },
    # Supabase integration
    "supabase": {
        "display_name": "Supabase",
        "oauth_provider": "supabase_mgmt",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/supabase.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#3ECF8E",
        "default_scopes": ["read:projects"],
        "scopes": [
            {
                "value": "read:projects",
                "display_name": "Read projects",
                "description": "View your Supabase projects"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["supabase_"],
            "scope_keywords": ["supabase"]
        }
    },
    # Slack integration
    "slack": {
        "display_name": "Slack",
        "oauth_provider": "slack",
        "requires_oauth": True,
        "icon": {"type": "url", "value": "https://img.logo.dev/slack.com?token=pk_Ovp4v2MYQDqQ0D6xdN7OJA"},
        "brand_color": "#4A154B",
        "default_scopes": ["channels:read", "chat:write"],
        "scopes": [
            {
                "value": "channels:read",
                "display_name": "View channels",
                "description": "View basic information about public channels in a workspace"
            },
            {
                "value": "channels:history",
                "display_name": "View channel messages",
                "description": "View messages and other content in public channels"
            },
            {
                "value": "chat:write",
                "display_name": "Send messages",
                "description": "Send messages as the app"
            },
            {
                "value": "channels:join",
                "display_name": "Join channels",
                "description": "Join public channels in the workspace"
            },
            {
                "value": "users:read",
                "display_name": "View users",
                "description": "View people in a workspace"
            },
            {
                "value": "groups:read",
                "display_name": "View private channels",
                "description": "View basic information about private channels"
            },
            {
                "value": "groups:history",
                "display_name": "View private channel messages",
                "description": "View messages and other content in private channels"
            },
            {
                "value": "im:write",
                "display_name": "Send direct messages",
                "description": "Start direct messages with people"
            },
            {
                "value": "reactions:write",
                "display_name": "Add reactions",
                "description": "Add emoji reactions to messages"
            },
        ],
        "detection_patterns": {
            "tool_name_patterns": ["slack_"],
            "scope_keywords": ["slack", "channels", "chat"]
        }
    },
}


def _collect_scopes_from_tools(integration_type: str) -> Set[str]:
    """
    Collect all unique scopes used by tools of a given integration type.

    Args:
        integration_type: The integration type to collect scopes for

    Returns:
        Set of unique scope strings
    """
    scopes: Set[str] = set()
    for tool in list_tools():
        if tool.integration_type == integration_type:
            scopes.update(tool.required_scopes or [])
    return scopes


def _build_provider_to_types_map() -> Dict[str, List[str]]:
    """
    Build mapping from OAuth providers to integration types.

    Returns:
        Dict mapping provider names to list of integration types
    """
    provider_map: Dict[str, Set[str]] = defaultdict(set)

    # From static config
    for int_type, config in INTEGRATION_CONFIGS.items():
        provider = config.get("oauth_provider")
        if provider:
            provider_map[provider].add(int_type)

    # From registered tools (catch any not in static config)
    for tool in list_tools():
        if tool.integration_type and tool.provider:
            provider_map[tool.provider].add(tool.integration_type)

    return {k: sorted(list(v)) for k, v in provider_map.items()}


def _get_integration_metadata(integration_type: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a single integration type.

    Args:
        integration_type: The integration type

    Returns:
        Integration metadata dict or None if not found
    """
    config = INTEGRATION_CONFIGS.get(integration_type)
    if not config:
        return None

    # Merge static config with dynamic scope discovery
    tool_scopes = _collect_scopes_from_tools(integration_type)
    configured_scope_values = {s["value"] for s in config.get("scopes", [])}

    # Add any scopes discovered from tools that aren't in static config
    additional_scopes = []
    for scope in tool_scopes:
        if scope not in configured_scope_values:
            # Auto-generate display name from scope
            display_name = _generate_scope_display_name(scope)
            additional_scopes.append({
                "value": scope,
                "display_name": display_name,
                "description": None
            })

    all_scopes = config.get("scopes", []) + additional_scopes

    return {
        "type": integration_type,
        "display_name": config["display_name"],
        "oauth_provider": config.get("oauth_provider"),
        "requires_oauth": config.get("requires_oauth", True),
        "icon": config["icon"],
        "brand_color": config.get("brand_color"),
        "default_scopes": config.get("default_scopes", []),
        "scopes": all_scopes,
        "detection_patterns": config.get("detection_patterns")
    }


def _generate_scope_display_name(scope: str) -> str:
    """
    Generate a human-readable display name from a scope string.

    Args:
        scope: OAuth scope string

    Returns:
        Human-readable display name
    """
    # Handle Google-style URLs
    if "googleapis.com/auth/" in scope:
        # Extract last part: "https://www.googleapis.com/auth/gmail.send" -> "gmail.send"
        name = scope.split("/auth/")[-1]
        # "gmail.send" -> "Gmail Send"
        name = name.replace(".", " ").replace("_", " ").title()
        return name

    # Handle simple scope names
    return scope.replace("_", " ").replace("-", " ").title()


def _discover_integration_types() -> Set[str]:
    """
    Discover all integration types from registered tools.

    Returns:
        Set of integration type strings
    """
    types: Set[str] = set()

    # From static config
    types.update(INTEGRATION_CONFIGS.keys())

    # From registered tools
    for tool in list_tools():
        if tool.integration_type:
            types.add(tool.integration_type)

    return types


def get_all_integration_metadata() -> Dict[str, Any]:
    """
    Get complete integration metadata for all integrations.

    This is the main function used by the API endpoint.

    Returns:
        Dict with 'integrations' list and 'provider_to_types' mapping
    """
    integrations = []
    all_types = _discover_integration_types()

    for int_type in sorted(all_types):
        metadata = _get_integration_metadata(int_type)
        if metadata:
            integrations.append(metadata)
        else:
            # Integration type from tools but not in config - create minimal entry
            logger.warning("Integration type '%s' has tools but no metadata config", int_type)
            integrations.append(_create_fallback_metadata(int_type))

    provider_to_types = _build_provider_to_types_map()

    return {
        "integrations": integrations,
        "provider_to_types": provider_to_types
    }


def _create_fallback_metadata(integration_type: str) -> Dict[str, Any]:
    """
    Create fallback metadata for an integration type without static config.

    Args:
        integration_type: The integration type

    Returns:
        Minimal integration metadata dict
    """
    # Try to determine provider from tools
    provider = None
    for tool in list_tools():
        if tool.integration_type == integration_type and tool.provider:
            provider = tool.provider
            break

    # Collect scopes from tools
    tool_scopes = _collect_scopes_from_tools(integration_type)
    scopes = [
        {"value": s, "display_name": _generate_scope_display_name(s), "description": None}
        for s in sorted(tool_scopes)
    ]

    return {
        "type": integration_type,
        "display_name": integration_type.replace("_", " ").title(),
        "oauth_provider": provider,
        "requires_oauth": provider is not None,
        "icon": {"type": "lucide", "value": "Wrench"},  # Default fallback icon
        "brand_color": None,
        "default_scopes": list(tool_scopes)[:3] if tool_scopes else [],
        "scopes": scopes,
        "detection_patterns": {
            "tool_name_patterns": [f"{integration_type}_"],
            "scope_keywords": [integration_type]
        }
    }


def get_integration_config(integration_type: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific integration type.

    Args:
        integration_type: The integration type to look up

    Returns:
        Integration config dict or None if not found
    """
    return INTEGRATION_CONFIGS.get(integration_type)


def get_display_name(integration_type: str) -> str:
    """
    Get display name for an integration type.

    Args:
        integration_type: The integration type

    Returns:
        Human-readable display name
    """
    config = INTEGRATION_CONFIGS.get(integration_type)
    if config:
        return config["display_name"]
    return integration_type.replace("_", " ").title()


def get_oauth_provider(integration_type: str) -> Optional[str]:
    """
    Get OAuth provider for an integration type.

    Args:
        integration_type: The integration type

    Returns:
        OAuth provider name or None
    """
    config = INTEGRATION_CONFIGS.get(integration_type)
    if config:
        return config.get("oauth_provider")
    return None
