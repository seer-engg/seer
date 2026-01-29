"""
Discord permission constants and bitfield management.

Discord uses numeric permission bitfields passed as 'permissions' query param.
Permissions are cumulative (bitwise OR).

References:
- Discord API Permissions: https://discord.com/developers/docs/topics/permissions
- Discord Permissions Calculator: https://discordapi.com/permissions.html
"""

from enum import IntFlag
from typing import List, Set


class DiscordPermission(IntFlag):
    """
    Discord permission bitfield constants.

    Each permission is represented as a bit in the permissions integer.
    Multiple permissions can be combined using bitwise OR (|).
    """

    # Basic channel permissions
    VIEW_CHANNEL = 1024  # 0x400 - View channels (including reading messages)
    SEND_MESSAGES = 2048  # 0x800 - Send messages in text channels
    MANAGE_CHANNELS = 16  # 0x10 - Manage channels (create, delete, edit)

    # Message content permissions
    EMBED_LINKS = 16384  # 0x4000 - Embed links in messages
    ATTACH_FILES = 32768  # 0x8000 - Attach files to messages
    READ_MESSAGE_HISTORY = 65536  # 0x10000 - Read message history
    ADD_REACTIONS = 64  # 0x40 - Add reactions to messages

    # Message management permissions
    MANAGE_MESSAGES = 8192  # 0x2000 - Delete and pin messages

    # Server management permissions
    MANAGE_ROLES = 268435456  # 0x10000000 - Manage roles
    MANAGE_WEBHOOKS = 536870912  # 0x20000000 - Manage webhooks
    CREATE_INSTANT_INVITE = 1  # 0x1 - Create instant invites

    # Member management permissions
    KICK_MEMBERS = 2  # 0x2 - Kick members from server
    BAN_MEMBERS = 4  # 0x4 - Ban members from server

    # Voice permissions (for future use)
    CONNECT = 1048576  # 0x100000 - Connect to voice channels
    SPEAK = 2097152  # 0x200000 - Speak in voice channels


# Tool-to-permission mapping
# Maps tool names to their required Discord bot permissions
TOOL_PERMISSIONS_MAP = {
    # Send channel message requires viewing the channel and sending messages
    "discord_send_channel_message": DiscordPermission.VIEW_CHANNEL | DiscordPermission.SEND_MESSAGES,
    # Send DM only requires sending messages (no channel viewing needed)
    "discord_send_direct_message": DiscordPermission.SEND_MESSAGES,
    # Find user requires viewing channels to see user list
    "discord_find_user": DiscordPermission.VIEW_CHANNEL,
}


def calculate_permissions(tool_names: List[str]) -> int:
    """
    Calculate minimal permission bitfield from tool names.

    Combines permissions from all specified tools using bitwise OR.
    Unknown tool names are silently ignored.

    Args:
        tool_names: List of tool names (e.g., ['discord_send_channel_message'])

    Returns:
        Combined permissions bitfield (int)

    Example:
        >>> calculate_permissions(['discord_send_channel_message'])
        3072  # VIEW_CHANNEL (1024) | SEND_MESSAGES (2048)
        >>> calculate_permissions(['discord_send_channel_message', 'discord_find_user'])
        3072  # Same as above since discord_find_user only needs VIEW_CHANNEL
    """
    permissions = 0
    for tool_name in tool_names:
        tool_perms = TOOL_PERMISSIONS_MAP.get(tool_name, 0)
        permissions |= tool_perms
    return permissions


def get_permission_names(permissions: int) -> Set[str]:
    """
    Convert permission bitfield to human-readable names.

    Args:
        permissions: Permission bitfield

    Returns:
        Set of permission names (e.g., {'VIEW_CHANNEL', 'SEND_MESSAGES'})

    Example:
        >>> get_permission_names(3072)
        {'VIEW_CHANNEL', 'SEND_MESSAGES'}
    """
    names = set()
    for perm in DiscordPermission:
        if permissions & perm:
            names.add(perm.name)
    return names


def has_permission(granted: int, required: int) -> bool:
    """
    Check if granted permissions satisfy required permissions.

    Uses bitwise AND to verify all required permission bits are set
    in the granted permissions.

    Args:
        granted: Granted permission bitfield
        required: Required permission bitfield

    Returns:
        True if all required permissions are present in granted

    Example:
        >>> has_permission(3072, 2048)  # Has SEND_MESSAGES?
        True
        >>> has_permission(1024, 2048)  # Has SEND_MESSAGES?
        False
    """
    return (granted & required) == required
