"""
Twilio tools — phone calls and WhatsApp messaging.

All Twilio tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.twilio.call import TwilioMakeCallTool
from seer.tools.twilio.send_whatsapp import TwilioSendWhatsAppTool


def register_twilio_tools():
    """Register all Twilio tools with the tool registry."""
    register_tool(TwilioMakeCallTool())
    register_tool(TwilioSendWhatsAppTool())


__all__ = [
    "register_twilio_tools",
    "TwilioMakeCallTool",
    "TwilioSendWhatsAppTool",
]
