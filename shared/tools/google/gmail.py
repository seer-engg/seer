from __future__ import annotations
"""
Gmail read tool using direct HTTP API calls.

Uses Gmail REST API with OAuth tokens - no google-auth library.
"""
"""
Gmail tools using direct HTTP API calls (no google-auth).

Implements: send, get, modify labels, trash/delete, threads, drafts, labels, attachments.

Official refs:
- Sending / raw MIME base64url: https://developers.google.com/workspace/gmail/api/guides/sending
- REST reference endpoints under: https://developers.google.com/workspace/gmail/api/reference/rest/v1/
"""


from typing import Any, Dict, List, Optional, Tuple, Union
import base64
import email.utils
from email.message import EmailMessage

import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool
from shared.logger import get_logger
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.gmail")


class GmailReadTool(BaseTool):
    """
    Tool for reading emails from Gmail inbox.
    
    Uses Gmail REST API v1 with direct HTTP calls.
    Requires OAuth scope: https://www.googleapis.com/auth/gmail.readonly
    """
    
    name = "gmail_read_emails"
    description = "Read emails from Gmail inbox. Supports filtering by labels, query, and max results."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for Gmail read tool parameters."""
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
        """
        Execute Gmail read tool.
        
        Args:
            access_token: OAuth access token (required)
            arguments: Tool arguments
        
        Returns:
            List of email objects with id, threadId, snippet, payload (subject, from, date)
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Gmail tool requires OAuth access token"
            )
        
        # Validate and convert max_results with defensive type checking
        max_results_raw = arguments.get("max_results", 10)
        if isinstance(max_results_raw, int):
            max_results = min(max_results_raw, 100)
        elif isinstance(max_results_raw, float) and max_results_raw.is_integer():
            max_results = min(int(max_results_raw), 100)
        elif isinstance(max_results_raw, str):
            try:
                max_results = min(int(max_results_raw), 100)
            except ValueError:
                logger.warning(f"Invalid max_results value '{max_results_raw}', using default 10")
                max_results = 10
        elif isinstance(max_results_raw, dict):
            # Try to extract numeric value from dict
            for key in ["value", "count", "output", "result", "number", "max_results"]:
                if key in max_results_raw:
                    nested_value = max_results_raw[key]
                    if isinstance(nested_value, (int, float)):
                        max_results = min(int(nested_value), 100)
                        break
                    elif isinstance(nested_value, str):
                        try:
                            max_results = min(int(nested_value), 100)
                            break
                        except ValueError:
                            continue
            else:
                logger.warning(f"Could not extract numeric value from max_results dict '{max_results_raw}', using default 10")
                max_results = 10
        else:
            logger.warning(f"Unexpected type for max_results: {type(max_results_raw).__name__}, using default 10")
            max_results = 10
        
        label_ids = arguments.get("label_ids", ["INBOX"])
        query = arguments.get("q")
        include_body = arguments.get("include_body", False)
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        # Build query parameters for listing messages
        params: Dict[str, Any] = {
            "maxResults": max_results
        }
        
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        
        if query:
            params["q"] = query
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: List messages
                logger.info(f"Fetching Gmail messages: max_results={max_results}, label_ids={label_ids}, q={query}")
                
                list_response = await client.get(
                    "https://www.googleapis.com/gmail/v1/users/me/messages",
                    headers=headers,
                    params=params
                )
                
                if list_response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="Gmail API authentication failed. Token may be expired or invalid."
                    )
                
                list_response.raise_for_status()
                list_data = list_response.json()
                
                messages = list_data.get("messages", [])
                if not messages:
                    logger.info("No messages found matching criteria")
                    return []
                
                logger.info(f"Found {len(messages)} messages, fetching details...")
                
                # Step 2: Fetch full message details
                results = []
                for msg in messages[:max_results]:
                    msg_id = msg["id"]
                    
                    # Build params for message detail
                    msg_params: Dict[str, Any] = {}
                    if include_body:
                        msg_params["format"] = "full"
                    else:
                        msg_params["format"] = "metadata"
                        msg_params["metadataHeaders"] = "From,To,Subject,Date"
                    
                    msg_response = await client.get(
                        f"https://www.googleapis.com/gmail/v1/users/me/messages/{msg_id}",
                        headers=headers,
                        params=msg_params
                    )
                    
                    if msg_response.status_code == 404:
                        logger.warning(f"Message {msg_id} not found, skipping")
                        continue
                    
                    msg_response.raise_for_status()
                    msg_data = msg_response.json()
                    
                    # Extract relevant fields
                    payload = msg_data.get("payload", {})
                    headers_list = payload.get("headers", [])
                    
                    # Convert headers list to dict for easier access
                    headers_dict = {h["name"]: h["value"] for h in headers_list}
                    
                    # Build email object
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
                    
                    # Include body if requested
                    if include_body:
                        # Extract body from payload
                        body_text = ""
                        if payload.get("body", {}).get("data"):
                            import base64
                            body_data = payload["body"]["data"]
                            body_text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                        elif payload.get("parts"):
                            # Multipart message
                            for part in payload["parts"]:
                                if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                                    import base64
                                    body_data = part["body"]["data"]
                                    body_text = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="ignore")
                                    break
                        
                        email_obj["body"] = body_text
                    
                    results.append(email_obj)
                
                logger.info(f"Successfully fetched {len(results)} email details")
                return results
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Gmail API error: {e.response.text[:500]}"
            )
        except httpx.TimeoutException:
            logger.error("Gmail API request timed out")
            raise HTTPException(
                status_code=504,
                detail="Gmail API request timed out"
            )
        except Exception as e:
            logger.exception(f"Unexpected error reading Gmail: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading Gmail: {str(e)}"
            )




logger = get_logger("shared.tools.gmail")

GMAIL_API_BASE = "https://www.googleapis.com/gmail/v1/users/me"


# -----------------------------
# Helpers (defensive coercion)
# -----------------------------
def _require_token(access_token: Optional[str]) -> str:
    if not access_token:
        raise HTTPException(status_code=401, detail="Gmail tool requires OAuth access token")
    return access_token


def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    """
    Robustly convert value to int with fallback.
    Accepts int/float(strict integer)/numeric str/dict with common numeric keys.
    """
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
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        # allow comma-separated
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    return default


def _b64url_encode(raw_bytes: bytes) -> str:
    # Gmail expects base64url; padding generally accepted but commonly stripped.
    return base64.urlsafe_b64encode(raw_bytes).decode("utf-8").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    # restore padding if stripped
    s = data.strip()
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _header_dict_from_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    headers_list = payload.get("headers", []) or []
    out: Dict[str, str] = {}
    for h in headers_list:
        name = h.get("name")
        value = h.get("value")
        if name and value is not None:
            out[str(name)] = str(value)
    return out


def _extract_text_body(payload: Dict[str, Any]) -> str:
    """
    Best-effort plain-text extraction for 'full' format messages.
    """
    # Single-part
    body = payload.get("body", {}) or {}
    data = body.get("data")
    if data:
        try:
            return _b64url_decode(data).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    # Multi-part
    parts = payload.get("parts", []) or []
    # Prefer text/plain
    for part in parts:
        if part.get("mimeType") == "text/plain":
            pdata = (part.get("body", {}) or {}).get("data")
            if pdata:
                try:
                    return _b64url_decode(pdata).decode("utf-8", errors="ignore")
                except Exception:
                    return ""
    # Fallback: first decodable part
    for part in parts:
        pdata = (part.get("body", {}) or {}).get("data")
        if pdata:
            try:
                return _b64url_decode(pdata).decode("utf-8", errors="ignore")
            except Exception:
                continue
    return ""


async def _gmail_request(
    access_token: str,
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    timeout_s: float = 30.0,
) -> httpx.Response:
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.request(method, url, headers=headers, params=params, json=json_body)
        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail="Gmail API authentication failed. Token may be expired or invalid.")
        return resp


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
    """
    Build an RFC 2822 MIME message. Gmail API requires this then base64url as Message.raw.
    """
    msg = EmailMessage()
    msg["To"] = ", ".join(to)
    msg["Subject"] = subject

    if cc:
        msg["Cc"] = ", ".join(cc)
    if bcc:
        # Bcc header is allowed; Gmail typically strips it for recipients, but safe to set.
        msg["Bcc"] = ", ".join(bcc)
    if from_email:
        msg["From"] = from_email
    if reply_to:
        msg["Reply-To"] = reply_to

    msg["Date"] = email.utils.formatdate(localtime=True)

    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references

    msg.set_content(body_text or "")

    if body_html:
        msg.add_alternative(body_html, subtype="html")

    # Attachments: expect base64 string in attachments[i]["data_base64"]
    # Optional: mime_type like "application/pdf"
    if attachments:
        for att in attachments:
            filename = str(att.get("filename") or "attachment")
            mime_type = str(att.get("mime_type") or "application/octet-stream")
            data_b64 = att.get("data_base64")
            if not data_b64:
                continue
            try:
                file_bytes = base64.b64decode(str(data_b64), validate=False)
            except Exception:
                # Try base64url as fallback
                try:
                    file_bytes = _b64url_decode(str(data_b64))
                except Exception:
                    logger.warning(f"Attachment '{filename}' has invalid base64; skipping")
                    continue

            if "/" in mime_type:
                maintype, subtype = mime_type.split("/", 1)
            else:
                maintype, subtype = "application", "octet-stream"

            msg.add_attachment(file_bytes, maintype=maintype, subtype=subtype, filename=filename)

    return msg


# -----------------------------
# Tools
# -----------------------------
class GmailSendEmailTool(BaseTool):
    """
    Send email via users.messages.send.
    """
    name = "gmail_send_email"
    description = "Send an email using Gmail. Supports plain text + optional HTML + optional attachments."
    required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "to": {"type": "array", "items": {"type": "string"}, "description": "Recipients (To)."},
                "subject": {"type": "string", "description": "Email subject."},
                "body_text": {"type": "string", "description": "Plain-text body."},
                "body_html": {"type": "string", "description": "Optional HTML body.", "default": None},
                "cc": {"type": "array", "items": {"type": "string"}, "description": "Optional CC list.", "default": []},
                "bcc": {"type": "array", "items": {"type": "string"}, "description": "Optional BCC list.", "default": []},
                "from_email": {"type": "string", "description": "Optional From (must be allowed alias).", "default": None},
                "reply_to": {"type": "string", "description": "Optional Reply-To.", "default": None},
                "thread_id": {"type": "string", "description": "Optional threadId to send in an existing thread.", "default": None},
                "in_reply_to": {"type": "string", "description": "Optional Message-ID for reply threading.", "default": None},
                "references": {"type": "string", "description": "Optional References header.", "default": None},
                "attachments": {
                    "type": "array",
                    "description": "Optional attachments. Each: {filename, mime_type, data_base64}.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "mime_type": {"type": "string"},
                            "data_base64": {"type": "string", "description": "Base64 (or base64url) encoded bytes."},
                        },
                        "required": ["data_base64"],
                    },
                    "default": [],
                },
            },
            "required": ["to", "subject", "body_text"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)

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
        body: Dict[str, Any] = {"raw": raw}
        if thread_id:
            body["threadId"] = str(thread_id)

        try:
            logger.info(f"Sending Gmail email to={to} subject='{subject[:80]}'")
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/messages/send", json_body=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail send error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            logger.exception("Unexpected error sending Gmail")
            raise HTTPException(status_code=500, detail=f"Error sending Gmail: {str(e)}")


class GmailGetMessageTool(BaseTool):
    """
    Get one message by ID via users.messages.get.
    """
    name = "gmail_get_message"
    description = "Get a Gmail message by ID. Supports format: minimal|metadata|full|raw."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Gmail message ID."},
                "format": {
                    "type": "string",
                    "description": "Response format: minimal|metadata|full|raw (default: metadata).",
                    "enum": ["minimal", "metadata", "full", "raw"],
                    "default": "metadata",
                },
                "metadata_headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If format=metadata, only return these headers (e.g. ['From','To','Subject','Date']).",
                    "default": ["From", "To", "Subject", "Date"],
                },
                "decode_body": {
                    "type": "boolean",
                    "description": "If format=full, extract best-effort plain-text body and return as 'body_text'.",
                    "default": False,
                },
            },
            "required": ["message_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)

        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")

        fmt = str(arguments.get("format") or "metadata")
        metadata_headers = _coerce_str_list(arguments.get("metadata_headers"), ["From", "To", "Subject", "Date"])
        decode_body = _coerce_bool(arguments.get("decode_body"), False)

        params: Dict[str, Any] = {"format": fmt}
        if fmt == "metadata" and metadata_headers:
            # Gmail expects repeated metadataHeaders params; comma-separated also works in practice in many clients.
            params["metadataHeaders"] = ",".join(metadata_headers)

        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/messages/{message_id}", params=params)
            resp.raise_for_status()
            data = resp.json()

            if decode_body and fmt == "full":
                payload = data.get("payload", {}) or {}
                data["body_text"] = _extract_text_body(payload)

            return data
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail get message error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            logger.exception("Unexpected error getting Gmail message")
            raise HTTPException(status_code=500, detail=f"Error getting Gmail message: {str(e)}")


class GmailModifyMessageLabelsTool(BaseTool):
    """
    Modify labels on a message (mark read/unread, archive, etc.) via users.messages.modify.
    """
    name = "gmail_modify_message_labels"
    description = "Add/remove labels on a Gmail message (e.g., mark read/unread, archive by removing INBOX)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Gmail message ID."},
                "add_label_ids": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Labels to add."},
                "remove_label_ids": {"type": "array", "items": {"type": "string"}, "default": [], "description": "Labels to remove."},
            },
            "required": ["message_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)

        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")

        add_label_ids = _coerce_str_list(arguments.get("add_label_ids"), [])
        remove_label_ids = _coerce_str_list(arguments.get("remove_label_ids"), [])

        body = {"addLabelIds": add_label_ids, "removeLabelIds": remove_label_ids}

        try:
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/messages/{message_id}/modify", json_body=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail modify labels error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            logger.exception("Unexpected error modifying Gmail labels")
            raise HTTPException(status_code=500, detail=f"Error modifying Gmail labels: {str(e)}")


class GmailTrashMessageTool(BaseTool):
    name = "gmail_trash_message"
    description = "Move a Gmail message to TRASH."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")
        try:
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/messages/{message_id}/trash")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error trashing Gmail message: {str(e)}")


class GmailDeleteMessageTool(BaseTool):
    name = "gmail_delete_message"
    description = "Permanently delete a Gmail message (cannot be undone; prefer trash)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"message_id": {"type": "string"}}, "required": ["message_id"]}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        message_id = str(arguments.get("message_id") or "").strip()
        if not message_id:
            raise HTTPException(status_code=400, detail="Parameter 'message_id' is required")
        try:
            resp = await _gmail_request(token, "DELETE", f"{GMAIL_API_BASE}/messages/{message_id}")
            resp.raise_for_status()
            # delete returns empty body on success for many endpoints; normalize
            return {"status": "deleted", "message_id": message_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting Gmail message: {str(e)}")


class GmailListThreadsTool(BaseTool):
    name = "gmail_list_threads"
    description = "List Gmail threads. Supports labelIds, q, maxResults, and pageToken."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "label_ids": {"type": "array", "items": {"type": "string"}, "default": ["INBOX"]},
                "q": {"type": "string", "default": None, "description": "Gmail search query to filter threads."},
                "page_token": {"type": "string", "default": None},
                "include_spam_trash": {"type": "boolean", "default": False},
            },
            "required": [],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)

        max_results = _coerce_int(arguments.get("max_results", 10), 10, min_value=1, max_value=100)
        label_ids = _coerce_str_list(arguments.get("label_ids"), ["INBOX"])
        q = arguments.get("q")
        page_token = arguments.get("page_token")
        include_spam_trash = _coerce_bool(arguments.get("include_spam_trash"), False)

        params: Dict[str, Any] = {"maxResults": max_results, "includeSpamTrash": include_spam_trash}
        if label_ids:
            params["labelIds"] = ",".join(label_ids)
        if q:
            params["q"] = str(q)
        if page_token:
            params["pageToken"] = str(page_token)

        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/threads", params=params)
            resp.raise_for_status()
            data = resp.json()
            return {
                "threads": data.get("threads", []) or [],
                "nextPageToken": data.get("nextPageToken"),
                "resultSizeEstimate": data.get("resultSizeEstimate"),
            }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing Gmail threads: {str(e)}")


class GmailGetThreadTool(BaseTool):
    name = "gmail_get_thread"
    description = "Get a Gmail thread by ID. Supports format: minimal|metadata|full."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "description": "Thread ID."},
                "format": {"type": "string", "enum": ["minimal", "metadata", "full"], "default": "metadata"},
                "metadata_headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["From", "To", "Subject", "Date"],
                },
            },
            "required": ["thread_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        thread_id = str(arguments.get("thread_id") or "").strip()
        if not thread_id:
            raise HTTPException(status_code=400, detail="Parameter 'thread_id' is required")

        fmt = str(arguments.get("format") or "metadata")
        metadata_headers = _coerce_str_list(arguments.get("metadata_headers"), ["From", "To", "Subject", "Date"])

        params: Dict[str, Any] = {"format": fmt}
        if fmt == "metadata" and metadata_headers:
            params["metadataHeaders"] = ",".join(metadata_headers)

        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/threads/{thread_id}", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting Gmail thread: {str(e)}")


class GmailCreateDraftTool(BaseTool):
    """
    Create a draft via users.drafts.create.
    """
    name = "gmail_create_draft"
    description = "Create a Gmail draft (DRAFT label). Supports plain text + optional HTML + optional attachments."
    required_scopes = ["https://www.googleapis.com/auth/gmail.compose"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return GmailSendEmailTool().get_parameters_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)

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

        try:
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/drafts", json_body=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Gmail create draft error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            logger.exception("Unexpected error creating Gmail draft")
            raise HTTPException(status_code=500, detail=f"Error creating Gmail draft: {str(e)}")


class GmailListDraftsTool(BaseTool):
    name = "gmail_list_drafts"
    description = "List Gmail drafts. Supports maxResults, q, pageToken, and includeSpamTrash."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

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
        token = _require_token(access_token)

        max_results = _coerce_int(arguments.get("max_results", 10), 10, min_value=1, max_value=100)
        q = arguments.get("q")
        page_token = arguments.get("page_token")
        include_spam_trash = _coerce_bool(arguments.get("include_spam_trash"), False)

        params: Dict[str, Any] = {"maxResults": max_results, "includeSpamTrash": include_spam_trash}
        if q:
            params["q"] = str(q)
        if page_token:
            params["pageToken"] = str(page_token)

        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/drafts", params=params)
            resp.raise_for_status()
            data = resp.json()
            return {"drafts": data.get("drafts", []) or [], "nextPageToken": data.get("nextPageToken")}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing Gmail drafts: {str(e)}")


class GmailGetDraftTool(BaseTool):
    name = "gmail_get_draft"
    description = "Get a Gmail draft by ID. Supports format: minimal|metadata|full|raw."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

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
        token = _require_token(access_token)
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")

        fmt = str(arguments.get("format") or "metadata")
        metadata_headers = _coerce_str_list(arguments.get("metadata_headers"), ["From", "To", "Subject", "Date"])

        params: Dict[str, Any] = {"format": fmt}
        if fmt == "metadata" and metadata_headers:
            params["metadataHeaders"] = ",".join(metadata_headers)

        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/drafts/{draft_id}", params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting Gmail draft: {str(e)}")


class GmailSendDraftTool(BaseTool):
    name = "gmail_send_draft"
    description = "Send a Gmail draft by ID (users.drafts.send). Optionally update raw content before sending."
    required_scopes = ["https://www.googleapis.com/auth/gmail.compose"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "draft_id": {"type": "string", "description": "Draft ID to send."},
                "update_raw_message": {
                    "type": "object",
                    "description": "Optional: if provided, update the MIME raw before sending. Same fields as send/create draft.",
                    "default": None,
                },
            },
            "required": ["draft_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")

        update = arguments.get("update_raw_message")

        body: Dict[str, Any] = {"id": draft_id}

        # If user wants to update content before sending
        if isinstance(update, dict):
            to = _coerce_str_list(update.get("to"), [])
            subject = str(update.get("subject") or "")
            body_text = str(update.get("body_text") or "")
            body_html = update.get("body_html")
            cc = _coerce_str_list(update.get("cc"), [])
            bcc = _coerce_str_list(update.get("bcc"), [])
            from_email = update.get("from_email")
            reply_to = update.get("reply_to")
            in_reply_to = update.get("in_reply_to")
            references = update.get("references")
            attachments = update.get("attachments") or []

            if to:
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
                body["message"] = {"raw": raw}

        try:
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/drafts/send", json_body=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error sending Gmail draft: {str(e)}")


class GmailDeleteDraftTool(BaseTool):
    name = "gmail_delete_draft"
    description = "Permanently delete a draft (users.drafts.delete)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.modify"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"draft_id": {"type": "string"}}, "required": ["draft_id"]}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        draft_id = str(arguments.get("draft_id") or "").strip()
        if not draft_id:
            raise HTTPException(status_code=400, detail="Parameter 'draft_id' is required")
        try:
            resp = await _gmail_request(token, "DELETE", f"{GMAIL_API_BASE}/drafts/{draft_id}")
            resp.raise_for_status()
            return {"status": "deleted", "draft_id": draft_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting Gmail draft: {str(e)}")


class GmailListLabelsTool(BaseTool):
    name = "gmail_list_labels"
    description = "List all labels in the user's mailbox."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        token = _require_token(access_token)
        try:
            resp = await _gmail_request(token, "GET", f"{GMAIL_API_BASE}/labels")
            resp.raise_for_status()
            return (resp.json().get("labels", []) or [])
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error listing Gmail labels: {str(e)}")


class GmailCreateLabelTool(BaseTool):
    name = "gmail_create_label"
    description = "Create a new Gmail label."
    required_scopes = ["https://www.googleapis.com/auth/gmail.labels"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Label display name."},
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
        token = _require_token(access_token)
        name = str(arguments.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="Parameter 'name' is required")

        body = {
            "name": name,
            "labelListVisibility": str(arguments.get("label_list_visibility") or "labelShow"),
            "messageListVisibility": str(arguments.get("message_list_visibility") or "show"),
        }

        try:
            resp = await _gmail_request(token, "POST", f"{GMAIL_API_BASE}/labels", json_body=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating Gmail label: {str(e)}")


class GmailDeleteLabelTool(BaseTool):
    name = "gmail_delete_label"
    description = "Permanently delete a Gmail label by ID."
    required_scopes = ["https://www.googleapis.com/auth/gmail.labels"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"label_id": {"type": "string"}}, "required": ["label_id"]}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        label_id = str(arguments.get("label_id") or "").strip()
        if not label_id:
            raise HTTPException(status_code=400, detail="Parameter 'label_id' is required")
        try:
            resp = await _gmail_request(token, "DELETE", f"{GMAIL_API_BASE}/labels/{label_id}")
            resp.raise_for_status()
            return {"status": "deleted", "label_id": label_id}
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting Gmail label: {str(e)}")


class GmailGetAttachmentTool(BaseTool):
    name = "gmail_get_attachment"
    description = "Get a message attachment by attachment ID (users.messages.attachments.get)."
    required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    integration_type = "gmail"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Message ID."},
                "attachment_id": {"type": "string", "description": "Attachment ID from message payload parts[].body.attachmentId."},
                "decode_bytes": {"type": "boolean", "default": False, "description": "If true, return decoded bytes as base64 (standard) in 'data_base64'."},
            },
            "required": ["message_id", "attachment_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Dict[str, Any]:
        token = _require_token(access_token)
        message_id = str(arguments.get("message_id") or "").strip()
        attachment_id = str(arguments.get("attachment_id") or "").strip()
        if not message_id or not attachment_id:
            raise HTTPException(status_code=400, detail="Parameters 'message_id' and 'attachment_id' are required")

        decode_bytes = _coerce_bool(arguments.get("decode_bytes"), False)

        try:
            resp = await _gmail_request(
                token,
                "GET",
                f"{GMAIL_API_BASE}/messages/{message_id}/attachments/{attachment_id}",
            )
            resp.raise_for_status()
            data = resp.json()  # typically { "size": int, "data": base64url }
            if decode_bytes and data.get("data"):
                raw_bytes = _b64url_decode(str(data["data"]))
                # Return as standard base64 for portability
                data["data_base64"] = base64.b64encode(raw_bytes).decode("utf-8")
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Gmail API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Gmail API request timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting Gmail attachment: {str(e)}")


