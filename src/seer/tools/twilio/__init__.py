"""
Twilio tools - making phone calls with TTS.

All Twilio tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.twilio.call import TwilioMakeCallTool


def register_twilio_tools():
    """Register all Twilio tools with the tool registry."""
    register_tool(TwilioMakeCallTool())


__all__ = [
    "register_twilio_tools",
    "TwilioMakeCallTool",
]
