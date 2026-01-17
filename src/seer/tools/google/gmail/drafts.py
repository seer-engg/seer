"""
Gmail draft operations - creating, listing, sending drafts.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gmail.helpers import (
    GMAIL_API_BASE,
    GMAIL_DRAFT_SCHEMA,
    GMAIL_MESSAGE_SCHEMA,
    _coerce_int,
    _coerce_bool,
    _coerce_str_list,
    _build_mime_email,
    _b64url_encode,
    _schema_with_defs,
    _schema_ref,
)

logger = get_logger("shared.tools.gmail.drafts")


class GmailCreateDraftTool(GoogleAPIClient):
    """Create a Gmail draft with full email formatting."""

    name = "gmail_create_draft"
    description = "Create a Gmail draft (DRAFT label). Supports plain text + optional HTML + optional attachments."
    required_scopes = ["https://www.googleapis.com/auth/gmail.compose"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_DRAFT_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipients (To)."},
                "subject": {"type": "string", "description": "Email subject."},
                "body_text": {"type": "string", "description": "Plain-text body."},
                "body_html": {"type": "string", "description": "Optional HTML body.", "default": None},
                "cc": {"type": "array", "items": {"type": "string"}, "default": []},
                "bcc": {"type": "array", "items": {"type": "string"}, "default": []},
                "from_email": {"type": "string", "default": None},
                "reply_to": {"type": "string", "default": None},
                "in_reply_to": {"type": "string", "default": None},
                "references": {"type": "string", "default": None},
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "data_base64": {"type": "string"},
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
        in_reply_to = arguments.get("in_reply_to")
        references = arguments.get("references")
        attachments = arguments.get("attachments") or []

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

        raw = _b64url_encode(mime_msg.as_bytes())
        body = {"message": {"raw": raw}}

        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/drafts",
            access_token,
            json_body=body
        )
        return resp.json()


class GmailListDraftsTool(GoogleAPIClient):
    """List Gmail drafts with filtering and pagination."""

    name = "gmail_list_drafts"
    description = "List Gmail drafts. Supports maxResults, q, pageToken, and includeSpamTrash."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return _schema_with_defs({
            "type": "object",
            "properties": {
                "drafts": {"type": "array", "items": _schema_ref("Draft")},
                "nextPageToken": {"type": ["string", "null"]},
                "resultSizeEstimate": {"type": ["integer", "null"]},
            },
            "required": ["drafts"],
            "additionalProperties": True,
        })

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "q": {"type": "string", "default": None, "description": "Gmail query to filter drafts."},
                "page_token": {"type": "string", "default": None},
                "include_spam_trash": {"type": "boolean", "default": False},
            },
            "required": [],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        max_results = _coerce_int(arguments.get("max_results", 10), 10, min_value=1, max_value=100)
        q = arguments.get("q")
        page_token = arguments.get("page_token")
        include_spam_trash = _coerce_bool(arguments.get("include_spam_trash"), False)

        params: Dict[str, Any] = {"maxResults": max_results, "includeSpamTrash": include_spam_trash}
        if q:
            params["q"] = str(q)
        if page_token:
            params["pageToken"] = str(page_token)

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/drafts",
            access_token,
            params=params
        )
        data = resp.json()
        return {"drafts": data.get("drafts", []) or [], "nextPageToken": data.get("nextPageToken")}


class GmailGetDraftTool(GoogleAPIClient):
    """Get a Gmail draft by ID with specified format."""

    name = "gmail_get_draft"
    description = "Get a Gmail draft by ID. Supports format: minimal|metadata|full|raw."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_DRAFT_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string", "description": "Draft ID."},
                "format": {"type": "string", "enum": ["minimal", "metadata", "full", "raw"], "default": "metadata"},
                "metadata_headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["From", "To", "Subject", "Date"],
                },
            },
            "required": ["draft_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")

        fmt = str(arguments.get("format") or "metadata")
        metadata_headers = _coerce_str_list(arguments.get("metadata_headers"), ["From", "To", "Subject", "Date"])

        params: Dict[str, Any] = {"format": fmt}
        if fmt == "metadata" and metadata_headers:
            params["metadataHeaders"] = ",".join(metadata_headers)

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/drafts/{draft_id}",
            access_token,
            params=params
        )
        return resp.json()


class GmailSendDraftTool(GoogleAPIClient):
    """Send a Gmail draft by ID, optionally updating content first."""

    name = "gmail_send_draft"
    description = "Send a Gmail draft by ID (users.drafts.send). Optionally update raw content before sending."
    required_scopes = ["https://www.googleapis.com/auth/gmail.compose"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string", "description": "Draft ID to send."},
                "update_raw_message": {
                    "type": "object",
                    "description": "Optional: if provided, update the MIME raw before sending.",
                    "default": None,
                },
            },
            "required": ["draft_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")

        update = arguments.get("update_raw_message")
        body: Dict[str, Any] = {"id": draft_id}

        # Update content if requested
        if isinstance(update, dict):
            to = _coerce_str_list(update.get("to"), [])
            if to:
                subject = str(update.get("subject") or "")
                body_text = str(update.get("body_text") or "")
                body_html = update.get("body_html")
                cc = _coerce_str_list(update.get("cc"), [])
                bcc = _coerce_str_list(update.get("bcc"), [])

                mime_msg = _build_mime_email(
                    to=to,
                    subject=subject,
                    body_text=body_text,
                    body_html=str(body_html) if body_html else None,
                    cc=cc,
                    bcc=bcc,
                )
                raw = _b64url_encode(mime_msg.as_bytes())
                body["message"] = {"raw": raw}

        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/drafts/send",
            access_token,
            json_body=body
        )
        return resp.json()


class GmailDeleteDraftTool(GoogleAPIClient):
    """Permanently delete a Gmail draft."""

    name = "gmail_delete_draft"
    description = "Permanently delete a draft (users.drafts.delete)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["deleted"]},
                "draft_id": {"type": "string"},
            },
            "required": ["status", "draft_id"],
            "additionalProperties": False,
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"draft_id": {"type": "string"}}, "required": ["draft_id"]}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")

        await self._make_request(
            "DELETE",
            f"{GMAIL_API_BASE}/drafts/{draft_id}",
            access_token
        )
        return {"status": "deleted", "draft_id": draft_id}
