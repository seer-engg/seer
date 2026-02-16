# pylint: disable=duplicate-code  # Reason: Gmail send and draft tools intentionally mirror schema and argument handling
"""
Gmail send operations - sending emails and replies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.core.files.resolver import FileResolutionError, resolve_file_input
from seer.core.files.schemas import FILE_INPUT_SCHEMA
from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gmail.helpers import (
    GMAIL_API_BASE,
    GMAIL_MESSAGE_SCHEMA,
    _coerce_str_list,
    _build_mime_email,
    _b64url_encode,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.gmail.send")


class GmailSendEmailTool(GoogleAPIClient):
    """Send email via Gmail API with attachments and HTML support."""

    name = "gmail_send_email"
    description = (
        "Send an email using Gmail. Supports plain text + optional HTML + optional attachments. "
        "Attachments can be file references from other tools or static files from user storage."
    )
    required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        # Build attachment schema that accepts file references
        attachment_file_schema = FILE_INPUT_SCHEMA.copy()
        attachment_file_schema["description"] = "File reference (from parent node or user storage)"

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
                    "description": "Optional attachments. Each item is a file reference from another tool or user storage.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file": attachment_file_schema,
                            "filename": {
                                "type": "string",
                                "description": "Override filename (uses original if not provided)"
                            },
                        },
                        "required": ["file"],
                    },
                    "default": [],
                },
            },
            "required": ["to", "subject", "body_text"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,  # pylint: disable=unused-argument  # Part of tool interface
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        to = _coerce_str_list(arguments.get("to"), [])
        if not to:
            raise HTTPException(status_code=400, detail="Parameter 'to' must be a non-empty list")

        # Resolve file attachments
        resolved_attachments = await self._resolve_attachments(arguments.get("attachments") or [], context)

        # Build MIME email with arguments passed directly
        mime_msg = _build_mime_email(
            to=to,
            subject=str(arguments.get("subject") or ""),
            body_text=str(arguments.get("body_text") or ""),
            body_html=str(arguments["body_html"]) if arguments.get("body_html") else None,
            cc=_coerce_str_list(arguments.get("cc"), []),
            bcc=_coerce_str_list(arguments.get("bcc"), []),
            from_email=str(arguments["from_email"]) if arguments.get("from_email") else None,
            reply_to=str(arguments["reply_to"]) if arguments.get("reply_to") else None,
            attachments=resolved_attachments,
            in_reply_to=str(arguments["in_reply_to"]) if arguments.get("in_reply_to") else None,
            references=str(arguments["references"]) if arguments.get("references") else None,
        )

        # Encode to base64url and build request body
        body: Dict[str, Any] = {"raw": _b64url_encode(mime_msg.as_bytes())}
        if arguments.get("thread_id"):
            body["threadId"] = str(arguments["thread_id"])

        logger.info("Sending Gmail email to=%s subject='%s' attachments=%d", to, arguments.get("subject", "")[:80], len(resolved_attachments))

        resp = await self._make_request("POST", f"{GMAIL_API_BASE}/messages/send", access_token, json_body=body)
        return resp.json()

    async def _resolve_attachments(
        self,
        raw_attachments: List[Dict[str, Any]],
        context: Optional["WorkflowRuntimeContext"],
    ) -> List[Dict[str, Any]]:
        """
        Resolve file references in attachments to actual file data.

        Args:
            raw_attachments: List of attachment dicts with 'file' field containing file refs.
            context: Workflow runtime context for file resolution.

        Returns:
            List of resolved attachments with 'filename', 'mime_type', and 'data_bytes' fields.
        """
        resolved = []
        for i, att in enumerate(raw_attachments):
            file_input = att.get("file")
            if not file_input:
                continue

            try:
                data_bytes, mime_type, original_filename = await resolve_file_input(file_input, context)
            except FileResolutionError as e:
                raise HTTPException(status_code=400, detail=f"Attachment {i}: {e}") from e
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Attachment {i}: {e}") from e

            # Use provided filename or fall back to resolved filename
            filename = att.get("filename") or original_filename or f"attachment_{i}"

            resolved.append({
                "filename": filename,
                "mime_type": mime_type,
                "data_bytes": data_bytes,
            })

        return resolved
