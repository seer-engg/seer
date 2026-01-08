"""
Common helpers and JSON schemas for Google Drive tools.
This module centralizes HTTP request handling and shared schemas used by
the read/write tool modules.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, List
import json
import uuid

import httpx
from fastapi import HTTPException
from shared.logger import get_logger

logger = get_logger("shared.tools.google_drive.common")


def _require_access_token(access_token: Optional[str], tool_name: str) -> None:
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=f"{tool_name} requires OAuth access token"
        )


def _http_exception_from_response(resp: httpx.Response, prefix: str) -> HTTPException:
    # Keep message bounded
    body_snippet = resp.text[:800] if resp.text else ""
    if resp.status_code == 401:
        return HTTPException(status_code=401, detail=f"{prefix}: authentication failed. Token may be expired/invalid.")
    if resp.status_code == 403:
        return HTTPException(status_code=403, detail=f"{prefix}: permission denied. Ensure access + correct OAuth scopes.")
    return HTTPException(status_code=resp.status_code, detail=f"{prefix}: {body_snippet}")


def _encode_multipart_related(metadata: Dict[str, Any], content_bytes: bytes, content_type: str) -> Dict[str, Any]:
    """
    Build multipart/related payload for Drive uploadType=multipart.
    Returns: { "body": bytes, "content_type": "multipart/related; boundary=..." }
    """
    boundary = f"==============={uuid.uuid4().hex}=="
    meta_json = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    parts: List[bytes] = []
    parts.append(
        b"--" + boundary.encode("utf-8") + b"\r\n"
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
        + meta_json + b"\r\n"
    )
    parts.append(
        b"--" + boundary.encode("utf-8") + b"\r\n"
        + f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        + content_bytes + b"\r\n"
    )
    parts.append(b"--" + boundary.encode("utf-8") + b"--\r\n")

    body = b"".join(parts)
    return {"body": body, "content_type": f"multipart/related; boundary={boundary}"}


# pylint: disable=too-many-arguments
async def _drive_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    content: Optional[bytes] = None,
    timeout: float = 30.0,
    prefix: str = "Google Drive request",
) -> httpx.Response:
    """
    Centralized HTTP requests to Google Drive with consistent
    timeout/exception handling and error-to-HTTPException translation.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            method_lower = method.lower()
            if method_lower == "get":
                resp = await http_client.get(url, headers=headers, params=params)
            elif method_lower == "post":
                resp = await http_client.post(url, headers=headers, params=params, json=json_body, content=content)
            elif method_lower == "patch":
                resp = await http_client.patch(url, headers=headers, params=params, json=json_body, content=content)
            elif method_lower == "delete":
                resp = await http_client.delete(url, headers=headers, params=params)
            else:
                resp = await http_client.request(method, url, headers=headers, params=params, json=json_body, content=content)

            if resp.is_error:
                raise _http_exception_from_response(resp, prefix)
            return resp
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail=f"{prefix} timed out") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected %s: %s", prefix, exc)
        raise HTTPException(status_code=500, detail=f"{prefix} error: {str(exc)}") from exc


# -----------------------------
# Output Schemas (JSON Schema)
# -----------------------------
def _drive_user_schema() -> Dict[str, Any]:
    # https://developers.google.com/workspace/drive/api/reference/rest/v3/User
    return {
        "type": "object",
        "properties": {
            "displayName": {"type": "string"},
            "kind": {"type": "string"},
            "me": {"type": "boolean"},
            "permissionId": {"type": "string"},
            "emailAddress": {"type": "string"},
            "photoLink": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _drive_file_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "mimeType": {"type": "string"},
            "parents": {"type": "array", "items": {"type": "string"}},
            "driveId": {"type": "string"},
            "createdTime": {"type": "string", "description": "RFC3339 timestamp"},
            "modifiedTime": {"type": "string", "description": "RFC3339 timestamp"},
            "size": {"type": "string", "description": "File size in bytes as a string (int64)"},
            "webViewLink": {"type": "string"},
            "webContentLink": {"type": "string"},
            "trashed": {"type": "boolean"},
            "owners": {"type": "array", "items": _drive_user_schema()},
        },
        "additionalProperties": True,
    }


def _drive_file_list_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "files": {"type": "array", "items": _drive_file_schema()},
            "nextPageToken": {"type": "string"},
            "kind": {"type": "string"},
            "incompleteSearch": {"type": "boolean"},
        },
        "required": ["files"],
        "additionalProperties": True,
    }


def _drive_permission_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "displayName": {"type": "string"},
            "type": {"type": "string"},
            "kind": {"type": "string"},
            "photoLink": {"type": "string"},
            "emailAddress": {"type": "string"},
            "role": {"type": "string"},
            "allowFileDiscovery": {"type": "boolean"},
            "domain": {"type": "string"},
            "expirationTime": {"type": "string"},
            "deleted": {"type": "boolean"},
            "view": {"type": "string"},
            "pendingOwner": {"type": "boolean"},
            "inheritedPermissionsDisabled": {"type": "boolean"},
            "permissionDetails": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "permissionType": {"type": "string"},
                        "inheritedFrom": {"type": "string"},
                        "role": {"type": "string"},
                        "inherited": {"type": "boolean"},
                    },
                    "additionalProperties": True,
                },
            },
            "teamDrivePermissionDetails": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
        },
        "additionalProperties": True,
    }


def _drive_about_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "user": _drive_user_schema(),
            "storageQuota": {
                "type": "object",
                "properties": {
                    "limit": {"type": "string"},
                    "usage": {"type": "string"},
                    "usageInDrive": {"type": "string"},
                    "usageInDriveTrash": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "importFormats": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "exportFormats": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "maxUploadSize": {"type": "string"},
            "canCreateDrives": {"type": "boolean"},
        },
        "additionalProperties": True,
    }


def _empty_object_schema(description: str = "Empty JSON object on success.") -> Dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {},
        "additionalProperties": True,
    }


def get_google_drive_common_attributes() -> Dict[str, str]:
    """Return common attributes for Google Drive tools."""
    return {
        "integration_type": "google_drive",
        "provider": "google"
    }


def get_file_id_resource_picker() -> Dict[str, Any]:
    """Return the standard resource picker configuration for file_id."""
    return {
        "file_id": {
            "resource_type": "google_drive_file",
            "display_field": "name",
            "value_field": "id",
            "search_enabled": True,
            "hierarchy": True,
        }
    }


def get_file_id_parameter_schema() -> Dict[str, Any]:
    """Return the standard parameter schema for file_id operations."""
    return {
        "type": "object",
        "properties": {
            "file_id": {"type": "string"},
        }
    }


def validate_permission_arguments(arguments: Dict[str, Any]) -> None:
    """Validate permission creation arguments."""
    file_id = arguments.get("file_id")
    p_type = arguments.get("type")
    role = arguments.get("role")

    if not file_id:
        raise HTTPException(status_code=400, detail="file_id is required")
    if not p_type:
        raise HTTPException(status_code=400, detail="type is required")
    if not role:
        raise HTTPException(status_code=400, detail="role is required")

    if p_type in ("user", "group") and not arguments.get("email_address"):
        raise HTTPException(status_code=400, detail="email_address is required for type=user/group")
    if p_type == "domain" and not arguments.get("domain"):
        raise HTTPException(status_code=400, detail="domain is required for type=domain")
