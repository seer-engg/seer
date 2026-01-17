"""
Gmail send operations - sending emails and replies.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gmail.helpers import (
    GMAIL_API_BASE,
    GMAIL_MESSAGE_SCHEMA,
    _coerce_str_list,
    _build_mime_email,
    _b64url_encode,
)

logger = get_logger("shared.tools.gmail.send")


class GmailSendEmailTool(GoogleAPIClient):
    """Send email via Gmail API with attachments and HTML support."""

    name = "gmail_send_email"
    description = "Send an email using Gmail. Supports plain text + optional HTML + optional attachments."
    required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recipients (To)."
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject."
                },
                "body_text": {
                    "type": "string",
                    "description": "Plain-text body."
                },
                "body_html": {
                    "type": "string",
                    "description": "Optional HTML body.",
                    "default": None
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional CC list.",
                    "default": []
                },
                "bcc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional BCC list.",
                    "default": []
                },
                "from_email": {
                    "type": "string",
                    "description": "Optional From (must be allowed alias).",
                    "default": None
                },
                "reply_to": {
                    "type": "string",
                    "description": "Optional Reply-To.",
                    "default": None
                },
                "thread_id": {
                    "type": "string",
                    "description": "Optional threadId to send in an existing thread.",
                    "default": None
                },
                "in_reply_to": {
                    "type": "string",
                    "description": "Optional Message-ID for reply threading.",
                    "default": None
                },
                "references": {
                    "type": "string",
                    "description": "Optional References header.",
                    "default": None
                },
                "attachments": {
                    "type": "array",
                    "description": "Optional attachments. Each: {filename, mime_type, data_base64}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "data_base64": {
                                "type": "string",
                                "description": "Base64 (or base64url) encoded bytes."
                            },
                        },
                        "required": ["data_base64"],
                    },
                    "default": [],
                },
            },
            "required": ["to", "subject", "body_text"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        to = _coerce_str_list(arguments.get("to"), [])
        if not to:
            raise HTTPException(status_code=400, detail="Parameter 'to' must be a non-empty list")

        subject = str(arguments.get("subject") or "")
        body_text = str(arguments.get("body_text") or "")
        body_html = arguments.get("body_html")
        cc = _coerce_str_list(arguments.get("cc"), [])
        bcc = _coerce_str_list(arguments.get("bcc"), [])
        from_email = arguments.get("from_email")
        reply_to = arguments.get("reply_to")
        thread_id = arguments.get("thread_id")
        in_reply_to = arguments.get("in_reply_to")
        references = arguments.get("references")
        attachments = arguments.get("attachments") or []

        # Build MIME email
        mime_msg = _build_mime_email(
            to=to,
            subject=subject,
            body_text=body_text,
            body_html=str(body_html) if body_html else None,
            cc=cc,
            bcc=bcc,
            from_email=str(from_email) if from_email else None,
            reply_to=str(reply_to) if reply_to else None,
            attachments=attachments if isinstance(attachments, list) else None,
            in_reply_to=str(in_reply_to) if in_reply_to else None,
            references=str(references) if references else None,
        )

        # Encode to base64url
        raw = _b64url_encode(mime_msg.as_bytes())
        body: Dict[str, Any] = {"raw": raw}
        if thread_id:
            body["threadId"] = str(thread_id)

        logger.info("Sending Gmail email to=%s subject='%s'", to, subject[:80])

        # Use base class HTTP client
        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/messages/send",
            access_token,
            json_body=body
        )

        return resp.json()
