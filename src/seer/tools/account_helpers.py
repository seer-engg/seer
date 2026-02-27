"""
Helper functions for OAuth account discovery in unified tools.

Extracted to reduce complexity and line count in unified_tools.py.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from seer.database.models_oauth import OAuthConnection


def check_scope_coverage(
    conn_scopes: Optional[str],
    required_scopes: list[str],
) -> tuple[bool, list[str]]:
    """
    Check if connection scopes cover the required scopes.

    Args:
        conn_scopes: Space or comma-separated scopes string from the connection
        required_scopes: List of scopes required by the tool/trigger

    Returns:
        Tuple of (has_required_scopes, missing_scopes_list)
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.services.integrations.auth.helpers import has_required_scopes, parse_scopes

    if not required_scopes:
        return True, []

    if not conn_scopes:
        return False, list(required_scopes)

    if has_required_scopes(conn_scopes, required_scopes):
        return True, []

    # Calculate which specific scopes are missing
    granted_set = parse_scopes(conn_scopes)
    missing = [scope for scope in required_scopes if scope not in granted_set]
    return False, missing


def build_account_entry(
    conn: "OAuthConnection",
    required_scopes: list[str],
) -> dict:
    """
    Build a standardized account entry dict for the accounts list.

    Args:
        conn: The OAuthConnection object
        required_scopes: List of scopes required by the tool/trigger

    Returns:
        Dict with id, provider_account_id, display_name, has_required_scopes, missing_scopes
    """
    # pylint: disable=import-outside-toplevel # Reason: Avoid circular imports
    from seer.services.integrations.auth.helpers import get_connection_display_name

    has_scopes, missing = check_scope_coverage(conn.scopes, required_scopes)

    return {
        "id": conn.id,
        "provider_account_id": conn.provider_account_id or f"ID:{conn.id}",
        "display_name": get_connection_display_name(conn),
        "has_required_scopes": has_scopes,
        "missing_scopes": missing,
    }


def make_error_response(
    key_name: str,
    key_value: str,
    error: str,
) -> dict:
    """
    Create a standardized error response for account lookup failures.

    Args:
        key_name: Either "tool_name" or "trigger_key"
        key_value: The actual tool name or trigger key
        error: Error message

    Returns:
        Dict with standard error structure
    """
    return {
        key_name: key_value,
        "provider": None,
        "accounts": [],
        "requires_selection": False,
        "error": error,
    }


def make_no_oauth_response(
    key_name: str,
    key_value: str,
    provider: Optional[str],
    message: str,
) -> dict:
    """
    Create a standardized response for tools/triggers that don't require OAuth.

    Args:
        key_name: Either "tool_name" or "trigger_key"
        key_value: The actual tool name or trigger key
        provider: The provider name (may be None)
        message: Informational message

    Returns:
        Dict with standard "no OAuth needed" structure
    """
    return {
        key_name: key_value,
        "provider": provider,
        "accounts": [],
        "requires_selection": False,
        "message": message,
    }
