"""
Google Drive helpers - multipart encoding and schema definitions.
"""

import json
import uuid
from typing import Any, Dict, List


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


# -----------------------------
# Drive API Schema Definitions
# -----------------------------

def _drive_user_schema() -> Dict[str, Any]:
    """User resource schema."""
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
    """File resource schema."""
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
    """FileList response schema."""
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
    """Permission resource schema."""
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
            "expirationTime": {"type": "string", "description": "RFC3339 timestamp"},
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
    """About resource schema."""
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "user": _drive_user_schema(),
            "storageQuota": {
                "type": "object",
                "properties": {
                    "limit": {"type": "string", "description": "int64 as string"},
                    "usage": {"type": "string", "description": "int64 as string"},
                    "usageInDrive": {"type": "string", "description": "int64 as string"},
                    "usageInDriveTrash": {"type": "string", "description": "int64 as string"},
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
            "maxUploadSize": {"type": "string", "description": "int64 as string"},
            "canCreateDrives": {"type": "boolean"},
        },
        "additionalProperties": True,
    }


def _empty_object_schema(description: str = "Empty JSON object on success.") -> Dict[str, Any]:
    """Empty success response schema."""
    return {
        "type": "object",
        "description": description,
        "properties": {},
        "additionalProperties": True,
    }
