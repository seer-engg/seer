"""
Twilio phone call tool — makes a call with TTS message.

Uses Twilio's Twiml parameter to speak a message without needing a webhook.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.twilio.base import TwilioAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.twilio.call")


class TwilioMakeCallTool(TwilioAPIClient):
    """Make a phone call with a text-to-speech message."""

    name = "twilio_make_call"
    description = "Make a phone call and speak a message using text-to-speech. The recipient hears the message when they pick up."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Phone number to call in E.164 format (e.g., +14155551234)",
                },
                "message": {
                    "type": "string",
                    "description": "Message to speak via text-to-speech when the call is answered",
                    "maxLength": 4096,
                },
            },
            "required": ["to", "message"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sid": {"type": "string", "description": "Twilio call SID"},
                "status": {"type": "string", "description": "Call status (queued, ringing, etc.)"},
                "to": {"type": "string", "description": "Called phone number"},
                "from": {"type": "string", "description": "Calling phone number"},
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

        logger.info("Making Twilio call to %s", to_number)

        twiml = f'<Response><Say voice="alice">{message}</Say></Response>'

        result = await self._make_request(
            "POST",
            f"/2010-04-01/Accounts/{account_sid}/Calls.json",
            credentials=credentials,
            data={
                "To": to_number,
                "From": from_number,
                "Twiml": twiml,
            },
        )

        return {
            "sid": result.get("sid"),
            "status": result.get("status"),
            "to": result.get("to"),
            "from": result.get("from"),
        }
