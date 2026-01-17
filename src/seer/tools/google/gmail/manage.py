"""
Gmail management operations - labels, trash, delete.
"""

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gmail.helpers import (
    GMAIL_API_BASE,
    GMAIL_MESSAGE_SCHEMA,
    GMAIL_LABEL_SCHEMA,
    _coerce_str_list,
    _schema_with_defs,
    _schema_ref,
)

logger = get_logger("shared.tools.gmail.manage")


class GmailModifyMessageLabelsTool(GoogleAPIClient):
    """Add/remove labels on a Gmail message."""

    name = "gmail_modify_message_labels"
    description = "Add/remove labels on a Gmail message (e.g., mark read/unread, archive by removing INBOX)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID."
                },
                "add_label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Labels to add."
                },
                "remove_label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Labels to remove."
                },
            },
            "required": ["message_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")

        add_label_ids = _coerce_str_list(arguments.get("add_label_ids"), [])
        remove_label_ids = _coerce_str_list(arguments.get("remove_label_ids"), [])

        body = {"addLabelIds": add_label_ids, "removeLabelIds": remove_label_ids}

        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/messages/{message_id}/modify",
            access_token,
            json_body=body
        )
        return resp.json()


class GmailTrashMessageTool(GoogleAPIClient):
    """Move a Gmail message to TRASH."""

    name = "gmail_trash_message"
    description = "Move a Gmail message to TRASH."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"}
            },
            "required": ["message_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")

        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/messages/{message_id}/trash",
            access_token
        )
        return resp.json()


class GmailDeleteMessageTool(GoogleAPIClient):
    """Permanently delete a Gmail message."""

    name = "gmail_delete_message"
    description = "Permanently delete a Gmail message (cannot be undone; prefer trash)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["deleted"]},
                "message_id": {"type": "string"},
            },
            "required": ["status", "message_id"],
            "additionalProperties": False,
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"}
            },
            "required": ["message_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")

        await self._make_request(
            "DELETE",
            f"{GMAIL_API_BASE}/messages/{message_id}",
            access_token
        )
        return {"status": "deleted", "message_id": message_id}


class GmailListLabelsTool(GoogleAPIClient):
    """List all labels in the user's mailbox."""

    name = "gmail_list_labels"
    description = "List all labels in the user's mailbox."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return _schema_with_defs({
            "type": "array",
            "items": _schema_ref("Label"),
        })

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/labels",
            access_token
        )
        return resp.json().get("labels", []) or []


class GmailCreateLabelTool(GoogleAPIClient):
    """Create a new Gmail label."""

    name = "gmail_create_label"
    description = "Create a new Gmail label."
    required_scopes = ["https://www.googleapis.com/auth/gmail.labels"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_LABEL_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Label display name."
                },
                "label_list_visibility": {
                    "type": "string",
                    "description": "Label list visibility.",
                    "default": "labelShow",
                },
                "message_list_visibility": {
                    "type": "string",
                    "description": "Message list visibility.",
                    "default": "show",
                },
            },
            "required": ["name"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        name = str(arguments.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Parameter 'name' is required")

        body = {
            "name": name,
            "labelListVisibility": str(arguments.get("label_list_visibility") or "labelShow"),
            "messageListVisibility": str(arguments.get("message_list_visibility") or "show"),
        }

        resp = await self._make_request(
            "POST",
            f"{GMAIL_API_BASE}/labels",
            access_token,
            json_body=body
        )
        return resp.json()


class GmailDeleteLabelTool(GoogleAPIClient):
    """Permanently delete a Gmail label by ID."""

    name = "gmail_delete_label"
    description = "Permanently delete a Gmail label by ID."
    required_scopes = ["https://www.googleapis.com/auth/gmail.labels"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["deleted"]},
                "label_id": {"type": "string"},
            },
            "required": ["status", "label_id"],
            "additionalProperties": False,
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label_id": {"type": "string"}
            },
            "required": ["label_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        label_id = str(arguments.get("label_id") or "").strip()
        if not label_id:
            raise HTTPException(status_code=400, detail="Parameter 'label_id' is required")

        await self._make_request(
            "DELETE",
            f"{GMAIL_API_BASE}/labels/{label_id}",
            access_token
        )
        return {"status": "deleted", "label_id": label_id}
