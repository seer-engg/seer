"""Runtime permission validation for Discord tools."""

from fastapi import HTTPException

from seer.database import OAuthConnection
from seer.logger import get_logger
from seer.tools.discord.permissions import has_permission, get_permission_names

logger = get_logger("shared.tools.discord.validators")


def validate_discord_permissions(
    connection: OAuthConnection,
    required_permissions: int,
    tool_name: str,
) -> None:
    """
    Validate that connection has required Discord bot permissions.

    Checks if the stored permission bitfield in connection metadata includes
    all required permissions for the tool.

    Args:
        connection: OAuthConnection instance for Discord
        required_permissions: Required permission bitfield for the tool
        tool_name: Tool name (for error messages)

    Raises:
        HTTPException: 403 if permissions are insufficient or missing

    Example:
        >>> validate_discord_permissions(
        ...     connection=discord_conn,
        ...     required_permissions=3072,  # VIEW_CHANNEL | SEND_MESSAGES
        ...     tool_name="discord_send_channel_message"
        ... )
    """
    if not connection.provider_metadata:
        logger.error(
            "%s: Connection %s has no provider_metadata, cannot verify permissions",
            tool_name,
            connection.id
        )
        raise HTTPException(
            status_code=403,
            detail=(
                f"{tool_name}: No permission metadata found. "
                f"Please reconnect your Discord integration to update permissions."
            )
        )

    granted_permissions = connection.provider_metadata.get("permissions", 0)

    if not has_permission(granted_permissions, required_permissions):
        # Calculate missing permissions
        missing_perms = required_permissions & ~granted_permissions
        missing_names = get_permission_names(missing_perms)
        required_names = get_permission_names(required_permissions)
        granted_names = get_permission_names(granted_permissions)

        logger.error(
            "%s: Insufficient Discord permissions. Required=%s (%s), Granted=%s (%s), Missing=%s (%s)",
            tool_name,
            required_permissions,
            required_names,
            granted_permissions,
            granted_names,
            missing_perms,
            missing_names
        )

        raise HTTPException(
            status_code=403,
            detail=(
                f"{tool_name}: Bot lacks required Discord permissions. "
                f"Missing: {', '.join(sorted(missing_names))}. "
                f"Please reconnect your Discord integration with the required permissions."
            )
        )

    logger.debug(
        "%s: Permission check passed. Required=%s, Granted=%s",
        tool_name,
        required_permissions,
        granted_permissions
    )
