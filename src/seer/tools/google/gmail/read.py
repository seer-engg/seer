"""
Gmail read operations - fetching emails, threads, and attachments.
"""

import base64
from typing import Any, Dict, List, Optional

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gmail.helpers import (
    GMAIL_API_BASE,
    GMAIL_MESSAGE_SCHEMA,
    GMAIL_THREAD_SCHEMA,
    GMAIL_ATTACHMENT_BODY_SCHEMA,
    _coerce_int,
    _b64url_decode,
    _header_dict_from_payload,
    _extract_text_body,
)

logger = get_logger("shared.tools.gmail.read")


class GmailReadTool(GoogleAPIClient):
    """Read emails from Gmail inbox with filtering and search."""

    name = "gmail_read_emails"
    description = "Read emails from Gmail inbox. Supports filtering by labels, query, and max results."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "threadId": {"type": "string"},
                    "snippet": {"type": "string"},
                    "subject": {"type": "string"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "date": {"type": "string"},
                    "labelIds": {"type": "array", "items": {"type": "string"}},
                    "body": {"type": "string", "description": "Present only when include_body=true."},
                },
                "required": ["id", "threadId", "snippet", "subject", "from", "to", "date", "labelIds"],
                "additionalProperties": True,
            },
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return (default: 10, max: 100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label IDs to filter by (e.g., ['INBOX', 'UNREAD'])",
                    "default": ["INBOX"]
                },
                "q": {
                    "type": "string",
                    "description": "Gmail search query (e.g., 'from:example@gmail.com', 'subject:meeting')",
                    "default": None
                },
                "include_body": {
                    "type": "boolean",
                    "description": "Whether to include full email body (default: false)",
                    "default": False
                }
            },
            "required": []
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        max_results = _coerce_int(arguments.get("max_results", 10), 10, min_value=1, max_value=100)
        label_ids = arguments.get("label_ids", ["INBOX"])
        query = arguments.get("q")
        include_body = arguments.get("include_body", False)

        params: Dict[str, Any] = {"maxResults": max_results}
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        if query:
            params["q"] = query

        logger.info("Fetching Gmail messages: max_results=%s, label_ids=%s, q=%s", max_results, label_ids, query)

        # List messages
        list_resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/messages",
            access_token,
            params=params
        )
        messages = list_resp.json().get("messages", [])

        if not messages:
            logger.info("No messages found matching criteria")
            return []

        logger.info("Found %s messages, fetching details...", len(messages))

        # Fetch message details
        msg_params = {"format": "full"} if include_body else {
            "format": "metadata",
            "metadataHeaders": "From,To,Subject,Date"
        }

        results = []
        for msg in messages[:max_results]:
            try:
                msg_resp = await self._make_request(
                    "GET",
                    f"{GMAIL_API_BASE}/messages/{msg['id']}",
                    access_token,
                    params=msg_params
                )
                msg_data = msg_resp.json()

                payload = msg_data.get("payload", {})
                headers_dict = _header_dict_from_payload(payload)

                email_obj = {
                    "id": msg_data.get("id"),
                    "threadId": msg_data.get("threadId"),
                    "snippet": msg_data.get("snippet", ""),
                    "subject": headers_dict.get("Subject", ""),
                    "from": headers_dict.get("From", ""),
                    "to": headers_dict.get("To", ""),
                    "date": headers_dict.get("Date", ""),
                    "labelIds": msg_data.get("labelIds", [])
                }

                if include_body:
                    email_obj["body"] = _extract_text_body(payload)

                results.append(email_obj)
            except Exception as e:
                logger.warning("Failed to fetch message %s: %s", msg['id'], e)
                continue

        logger.info("Successfully fetched %s email details", len(results))
        return results


class GmailGetMessageTool(GoogleAPIClient):
    """Get a single Gmail message by ID with full details."""

    name = "gmail_get_message"
    description = "Get a single Gmail message by ID. Returns full message details including headers and body."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_MESSAGE_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID to retrieve"
                },
                "format": {
                    "type": "string",
                    "description": "Message format: 'full', 'metadata', 'minimal', or 'raw'",
                    "enum": ["full", "metadata", "minimal", "raw"],
                    "default": "full"
                }
            },
            "required": ["message_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        message_id = arguments.get("message_id")
        if not message_id:
            raise ValueError("message_id is required")

        msg_format = arguments.get("format", "full")
        params = {"format": msg_format}

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/messages/{message_id}",
            access_token,
            params=params
        )
        return resp.json()


class GmailListThreadsTool(GoogleAPIClient):
    """List Gmail threads with optional filtering."""

    name = "gmail_list_threads"
    description = "List Gmail threads. Supports filtering by labels and search queries."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "threads": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"id": {"type": "string"}, "snippet": {"type": "string"}}}
                },
                "nextPageToken": {"type": "string"},
                "resultSizeEstimate": {"type": "integer"}
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_results": {
                    "type": "integer",
                    "description": "Maximum threads to return (default: 10, max: 100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "label_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by label IDs"
                },
                "q": {
                    "type": "string",
                    "description": "Gmail search query"
                }
            },
            "required": []
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        max_results = _coerce_int(arguments.get("max_results", 10), 10, min_value=1, max_value=100)
        label_ids = arguments.get("label_ids")
        query = arguments.get("q")

        params: Dict[str, Any] = {"maxResults": max_results}
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        if query:
            params["q"] = query

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/threads",
            access_token,
            params=params
        )
        return resp.json()


class GmailGetThreadTool(GoogleAPIClient):
    """Get a complete Gmail thread by ID."""

    name = "gmail_get_thread"
    description = "Get a Gmail thread by ID. Returns all messages in the thread."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_THREAD_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "Gmail thread ID to retrieve"
                },
                "format": {
                    "type": "string",
                    "description": "Message format: 'full', 'metadata', 'minimal'",
                    "enum": ["full", "metadata", "minimal"],
                    "default": "full"
                }
            },
            "required": ["thread_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        thread_id = arguments.get("thread_id")
        if not thread_id:
            raise ValueError("thread_id is required")

        msg_format = arguments.get("format", "full")
        params = {"format": msg_format}

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/threads/{thread_id}",
            access_token,
            params=params
        )
        return resp.json()


class GmailGetAttachmentTool(GoogleAPIClient):
    """Get a Gmail message attachment by message ID and attachment ID."""

    name = "gmail_get_attachment"
    description = "Get a Gmail message attachment. Returns base64-encoded attachment data."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"

    def get_output_schema(self) -> Dict[str, Any]:
        return GMAIL_ATTACHMENT_BODY_SCHEMA

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "Gmail message ID containing the attachment"
                },
                "attachment_id": {
                    "type": "string",
                    "description": "Attachment ID to retrieve"
                },
                "decode_bytes": {
                    "type": "boolean",
                    "description": "If true, decode and return as standard base64 (default: false)",
                    "default": False
                }
            },
            "required": ["message_id", "attachment_id"]
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        message_id = arguments.get("message_id")
        attachment_id = arguments.get("attachment_id")
        decode_bytes = arguments.get("decode_bytes", False)

        if not message_id or not attachment_id:
            raise ValueError("message_id and attachment_id are required")

        resp = await self._make_request(
            "GET",
            f"{GMAIL_API_BASE}/messages/{message_id}/attachments/{attachment_id}",
            access_token
        )
        data = resp.json()

        if decode_bytes and data.get("data"):
            raw_bytes = _b64url_decode(str(data["data"]))
            data["data_base64"] = base64.b64encode(raw_bytes).decode("utf-8")

        return data
