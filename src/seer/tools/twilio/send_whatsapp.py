"""
Twilio WhatsApp tool — sends a WhatsApp message via Twilio Messages API.

Uses the `whatsapp:` prefix on From/To numbers per Twilio's WhatsApp API.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.twilio.base import TwilioAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.twilio.send_whatsapp")


class TwilioSendWhatsAppTool(TwilioAPIClient):
    """Send a WhatsApp message via Twilio."""

    name = "twilio_send_whatsapp"
    integration_type = "whatsapp"
    description = "Send a WhatsApp message to a phone number. The recipient receives it as a WhatsApp message from your Twilio number."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Phone number to send WhatsApp message to in E.164 format (e.g., +14155551234)",
                },
                "message": {
                    "type": "string",
                    "description": "Text body of the WhatsApp message",
                    "maxLength": 4096,
                },
            },
            "required": ["to", "message"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sid": {"type": "string", "description": "Twilio message SID"},
                "status": {"type": "string", "description": "Message status (queued, sent, etc.)"},
                "to": {"type": "string", "description": "Recipient WhatsApp number"},
                "from": {"type": "string", "description": "Sender WhatsApp number"},
            },
            "required": ["sid", "status"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context
        to_number = arguments.get("to")
        message = arguments.get("message")

        if not to_number:
            raise HTTPException(status_code=400, detail="Parameter 'to' is required")
        if not message:
            raise HTTPException(status_code=400, detail="Parameter 'message' is required")

        account_sid, _, from_number = self._get_credentials(credentials)

        # Prefer a WhatsApp-specific from-number (sandbox number differs from voice number)
        wa_from = None
        if credentials and credentials.secrets:
            wa_from = credentials.secrets.get("twilio_whatsapp_from_number")
        if not wa_from:
            from seer.config import config as _cfg  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports
            wa_from = _cfg.twilio_whatsapp_from_number
        if wa_from:
            from_number = wa_from
        else:
            raise HTTPException(
                status_code=400,
                detail="twilio_send_whatsapp: No WhatsApp from-number configured. "
                       "Set twilio_whatsapp_from_number in your Twilio integration or app config.",
            )

        # Strip whatsapp: prefix if caller already included it
        to_clean = to_number.removeprefix("whatsapp:")
        from_clean = from_number.removeprefix("whatsapp:")

        logger.info("Sending WhatsApp message to %s", to_clean)

        result = await self._make_request(
            "POST",
            f"/2010-04-01/Accounts/{account_sid}/Messages.json",
            credentials=credentials,
            data={
                "To": f"whatsapp:{to_clean}",
                "From": f"whatsapp:{from_clean}",
                "Body": message,
            },
        )

        return {
            "sid": result.get("sid"),
            "status": result.get("status"),
            "to": result.get("to"),
            "from": result.get("from"),
        }
