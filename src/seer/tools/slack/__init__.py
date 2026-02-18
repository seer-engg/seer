"""
Slack tools - sending messages, listing channels/users, and reactions.

All Slack tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.slack.messages import (
    SlackSendChannelMessageTool,
    SlackSendDirectMessageTool,
)
from seer.tools.slack.channels import (
    SlackListChannelsTool,
    SlackGetChannelHistoryTool,
    SlackJoinChannelTool,
)
from seer.tools.slack.users import SlackListUsersTool
from seer.tools.slack.reactions import SlackAddReactionTool


def register_slack_tools():
    """Register all Slack tools with the tool registry."""
    register_tool(SlackSendChannelMessageTool())
    register_tool(SlackSendDirectMessageTool())
    register_tool(SlackListChannelsTool())
    register_tool(SlackListUsersTool())
    register_tool(SlackGetChannelHistoryTool())
    register_tool(SlackAddReactionTool())
    register_tool(SlackJoinChannelTool())


__all__ = [
    "register_slack_tools",
    "SlackSendChannelMessageTool",
    "SlackSendDirectMessageTool",
    "SlackListChannelsTool",
    "SlackListUsersTool",
    "SlackGetChannelHistoryTool",
    "SlackAddReactionTool",
    "SlackJoinChannelTool",
]
