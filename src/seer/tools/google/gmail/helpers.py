"""
Shared helpers and utilities for Gmail tools.

Contains:
- Base64url encoding/decoding
- Header sanitization
- MIME message building
- Gmail API schema definitions
"""

import base64
import copy
import email.utils
import json
import re
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger

logger = get_logger("shared.tools.gmail.helpers")

# Gmail API base URL
GMAIL_API_BASE = "https://www.googleapis.com/gmail/v1/users/me"


# -----------------------------
# Type Coercion (Legacy - TODO: Replace with Pydantic)
# -----------------------------

def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    """Robustly convert value to int with fallback."""
    try:
        if isinstance(value, int):
            return max(min_value, min(value, max_value))
        if isinstance(value, float) and value.is_integer():
            return max(min_value, min(int(value), max_value))
        if isinstance(value, str):
            return max(min_value, min(int(value.strip()), max_value))
        if isinstance(value, dict):
            for key in ["value", "count", "output", "result", "number", "max_results", "maxResults"]:
                if key in value and isinstance(value[key], (int, float, str)):
                    return _coerce_int(value[key], default, min_value=min_value, max_value=max_value)
    except Exception:
        pass
    return max(min_value, min(default, max_value))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["true", "1", "yes", "y", "on"]:
            return True
        if v in ["false", "0", "no", "n", "off"]:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_str_list(value: Any, default: List[str]) -> List[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.startswith("[") and trimmed.endswith("]"):
            try:
                parsed = json.loads(trimmed)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                inner = trimmed[1:-1].strip()
                if inner:
                    trimmed = inner
                else:
                    return default
        parts = [p.strip() for p in trimmed.split(",")]
        return [p for p in parts if p]
    return default


# -----------------------------
# Base64url Encoding/Decoding
# -----------------------------

def _b64url_encode(raw_bytes: bytes) -> str:
    """Encode bytes to base64url format (Gmail API standard)."""
    return base64.urlsafe_b64encode(raw_bytes).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    """Decode base64url data (restores padding if stripped)."""
    s = data.strip()
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)


# -----------------------------
# Header and Body Parsing
# -----------------------------

_HEADER_SANITIZE_RE = re.compile(r"[\r\n]+")


def _sanitize_header_value(value: Optional[str]) -> Optional[str]:
    """Sanitize header value by removing newlines."""
    if value is None:
        return None
    cleaned = _HEADER_SANITIZE_RE.sub(" ", str(value)).strip()
    return cleaned or None


def _sanitize_address_list(addresses: Optional[List[str]]) -> List[str]:
    """Sanitize list of email addresses."""
    if not addresses:
        return []
    sanitized: List[str] = []
    for addr in addresses:
        cleaned = _sanitize_header_value(addr)
        if cleaned:
            sanitized.append(cleaned)
    return sanitized


def _header_dict_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    """Extract headers from Gmail message payload."""
    headers_list = payload.get("headers", []) or []
    out: Dict[str, str] = {}
    for h in headers_list:
        name = h.get("name")
        value = h.get("value")
        if name and value is not None:
            out[str(name)] = str(value)
    return out


def _decode_body_data(data: Optional[str]) -> str:
    """Decode base64url body data."""
    if not data:
        return ""
    try:
        return _b64url_decode(data).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_part_data(part: Dict[str, Any]) -> str:
    """Extract data from a single message part."""
    pdata = (part.get("body", {}) or {}).get("data")
    return _decode_body_data(pdata)


def _extract_text_body(payload: Dict[str, Any]) -> str:
    """Best-effort plain-text extraction for 'full' format messages."""
    body = payload.get("body", {}) or {}
    data = body.get("data")
    if data:
        return _decode_body_data(data)

    parts = payload.get("parts", []) or []

    for part in parts:
        if part.get("mimeType") == "text/plain":
            result = _extract_part_data(part)
            if result:
                return result

    for part in parts:
        result = _extract_part_data(part)
        if result:
            return result

    return ""


# -----------------------------
# MIME Message Building
# -----------------------------

def _parse_mime_type(mime_type: str) -> tuple[str, str]:
    """Parse MIME type into maintype and subtype."""
    if "/" in mime_type:
        return mime_type.split("/", 1)
    return "application", "octet-stream"


def _decode_attachment_data(data_b64: str, filename: str) -> Optional[bytes]:
    """Decode attachment data from base64."""
    try:
        return base64.b64decode(str(data_b64), validate=False)
    except Exception:
        try:
            return _b64url_decode(str(data_b64))
        except Exception:
            logger.warning("Attachment '%s' has invalid base64; skipping", filename)
            return None


def _add_attachments(msg: EmailMessage, attachments: Optional[List[Dict[str, Any]]]):
    """Add attachments to email message."""
    if not attachments:
        return

    for att in attachments:
        filename = str(att.get("filename") or "attachment")
        mime_type = str(att.get("mime_type") or "application/octet-stream")
        data_b64 = att.get("data_base64")

        if not data_b64:
            continue

        file_bytes = _decode_attachment_data(data_b64, filename)
        if not file_bytes:
            continue

        maintype, subtype = _parse_mime_type(mime_type)
        msg.add_attachment(file_bytes, maintype=maintype, subtype=subtype, filename=filename)


def _set_optional_header(msg: EmailMessage, name: str, value: Optional[str]):
    """Set optional email header if value is provided."""
    sanitized = _sanitize_header_value(value)
    if sanitized:
        msg[name] = sanitized


def _set_optional_addresses(msg: EmailMessage, name: str, addresses: Optional[List[str]]):
    """Set optional address list header if addresses are provided."""
    sanitized = _sanitize_address_list(addresses)
    if sanitized:
        msg[name] = ", ".join(sanitized)


def _build_mime_email(
    *,
    to: List[str],
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
    cc: Optional[List[str]] = None,
    bcc: Optional[List[str]] = None,
    from_email: Optional[str] = None,
    reply_to: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    in_reply_to: Optional[str] = None,
    references: Optional[str] = None,
) -> EmailMessage:
    """Build an RFC 2822 MIME message for Gmail API."""
    msg = EmailMessage()

    sanitized_to = _sanitize_address_list(to)
    if not sanitized_to:
        raise HTTPException(status_code=400, detail="Parameter 'to' must contain at least one valid address")

    msg["To"] = ", ".join(sanitized_to)
    msg["Subject"] = _sanitize_header_value(subject) or ""
    msg["Date"] = email.utils.formatdate(localtime=True)

    _set_optional_addresses(msg, "Cc", cc)
    _set_optional_addresses(msg, "Bcc", bcc)
    _set_optional_header(msg, "From", from_email)
    _set_optional_header(msg, "Reply-To", reply_to)
    _set_optional_header(msg, "In-Reply-To", in_reply_to)
    _set_optional_header(msg, "References", references)

    msg.set_content(body_text or "")
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    _add_attachments(msg, attachments)

    return msg


# -----------------------------
# Gmail API Schema Definitions
# -----------------------------

GMAIL_SCHEMA_DEFINITIONS: Dict[str, Any] = {
    "Header": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["name", "value"],
        "additionalProperties": True,
    },
    "MessagePartBody": {
        "type": "object",
        "properties": {
            "attachmentId": {"type": "string"},
            "size": {"type": "integer"},
            "data": {"type": "string", "description": "Base64url-encoded body data (may be omitted)."},
        },
        "required": [],
        "additionalProperties": True,
    },
    "AttachmentBody": {
        "type": "object",
        "properties": {
            "attachmentId": {"type": "string"},
            "size": {"type": "integer"},
            "data": {"type": "string", "description": "Base64url-encoded body data (may be omitted)."},
            "data_base64": {
                "type": "string",
                "description": "Added by tool when decode_bytes=true (standard base64).",
            },
        },
        "required": [],
        "additionalProperties": True,
    },
    "MessagePart": {
        "type": "object",
        "properties": {
            "partId": {"type": "string"},
            "mimeType": {"type": "string"},
            "filename": {"type": "string"},
            "headers": {"type": "array", "items": {"$ref": "#/$defs/Header"}},
            "body": {"$ref": "#/$defs/MessagePartBody"},
            "parts": {
                "type": "array",
                "items": {"$ref": "#/$defs/MessagePart"},
            },
        },
        "required": [],
        "additionalProperties": True,
    },
    "Message": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "threadId": {"type": "string"},
            "labelIds": {"type": "array", "items": {"type": "string"}},
            "snippet": {"type": "string"},
            "historyId": {"type": "string"},
            "internalDate": {"type": "string", "description": "Epoch ms as string."},
            "payload": {"$ref": "#/$defs/MessagePart"},
            "sizeEstimate": {"type": "integer"},
            "raw": {"type": "string", "description": "Base64url-encoded RFC 2822 message (format=raw)."},
        },
        "required": [],
        "additionalProperties": True,
    },
    "MessageIdOnly": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "threadId": {"type": "string"},
        },
        "required": ["id", "threadId"],
        "additionalProperties": True,
    },
    "Thread": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "snippet": {"type": "string"},
            "historyId": {"type": "string"},
            "messages": {"type": "array", "items": {"$ref": "#/$defs/Message"}},
        },
        "required": [],
        "additionalProperties": True,
    },
    "Draft": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "message": {"$ref": "#/$defs/Message"},
        },
        "required": [],
        "additionalProperties": True,
    },
    "LabelColor": {
        "type": "object",
        "properties": {
            "backgroundColor": {"type": "string"},
            "textColor": {"type": "string"},
        },
        "required": [],
        "additionalProperties": True,
    },
    "Label": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "messageListVisibility": {"type": "string"},
            "labelListVisibility": {"type": "string"},
            "type": {"type": "string"},
            "messagesTotal": {"type": "integer"},
            "messagesUnread": {"type": "integer"},
            "threadsTotal": {"type": "integer"},
            "threadsUnread": {"type": "integer"},
            "color": {"$ref": "#/$defs/LabelColor"},
        },
        "required": [],
        "additionalProperties": True,
    },
}


def _schema_with_defs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add schema definitions to a schema document."""
    document = copy.deepcopy(schema)
    document["$defs"] = copy.deepcopy(GMAIL_SCHEMA_DEFINITIONS)
    return document


def _schema_ref(root_name: str) -> Dict[str, Any]:
    """Create a schema reference."""
    return {"$ref": f"#/$defs/{root_name}"}


# Pre-built schemas with definitions
GMAIL_MESSAGE_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("Message"))
GMAIL_MESSAGE_ID_ONLY_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("MessageIdOnly"))
GMAIL_THREAD_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("Thread"))
GMAIL_DRAFT_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("Draft"))
GMAIL_LABEL_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("Label"))
GMAIL_ATTACHMENT_BODY_SCHEMA: Dict[str, Any] = _schema_with_defs(_schema_ref("AttachmentBody"))
