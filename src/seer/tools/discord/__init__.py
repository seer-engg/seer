"""
Discord tools - sending messages and finding users.

All Discord tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.discord.messages import (
    DiscordSendChannelMessageTool,
    DiscordSendDirectMessageTool,
)
from seer.tools.discord.users import DiscordFindUserTool


def register_discord_tools():
    """Register all Discord tools with the tool registry."""
    register_tool(DiscordSendChannelMessageTool())
    register_tool(DiscordSendDirectMessageTool())
    register_tool(DiscordFindUserTool())


__all__ = [
    "register_discord_tools",
    "DiscordSendChannelMessageTool",
    "DiscordSendDirectMessageTool",
    "DiscordFindUserTool",
]
