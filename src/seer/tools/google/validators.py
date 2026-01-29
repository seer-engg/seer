"""Runtime scope validation for Google tools."""

from typing import List

from fastapi import HTTPException

from seer.database import OAuthConnection
from seer.logger import get_logger
from seer.services.integrations.auth.helpers import has_required_scopes

logger = get_logger("shared.tools.google.validators")


def validate_google_scopes(
    connection: OAuthConnection,
    required_scopes: List[str],
    tool_name: str,
) -> None:
    """
    Validate that connection has required Google OAuth scopes.

    Checks if the stored scopes in connection include all required scopes
    for the tool, considering Google's scope hierarchy (e.g., gmail satisfies gmail.readonly).

    Args:
        connection: OAuthConnection instance for Google
        required_scopes: List of required OAuth scope URLs for the tool
        tool_name: Tool name (for error messages)

    Raises:
        HTTPException: 403 if scopes are insufficient or missing

    Example:
        >>> validate_google_scopes(
        ...     connection=google_conn,
        ...     required_scopes=["https://www.googleapis.com/auth/gmail.send"],
        ...     tool_name="gmail_send_email"
        ... )
    """
    granted_scopes = connection.scopes or ""

    if not has_required_scopes(granted_scopes, required_scopes):
        # Calculate missing scopes
        missing_scopes = [
            scope for scope in required_scopes
            if not has_required_scopes(granted_scopes, [scope])
        ]

        logger.error(
            "%s: Missing required Google OAuth scopes. Required=%s, Granted=%s..., Missing=%s",
            tool_name,
            required_scopes,
            granted_scopes[:100],  # Truncate for logging
            missing_scopes
        )

        raise HTTPException(
            status_code=403,
            detail=(
                f"{tool_name}: Missing required Google OAuth scopes. "
                f"Missing: {', '.join(missing_scopes)}. "
                f"Please reconnect your Google integration with the required scopes."
            )
        )

    logger.debug(
        "%s: Scope check passed. Required=%s",
        tool_name,
        required_scopes
    )
