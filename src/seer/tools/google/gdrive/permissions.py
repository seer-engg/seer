"""
Google Drive permission operations - sharing and account info.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gdrive.helpers import (
    _drive_permission_schema,
    _drive_about_schema,
)

logger = get_logger("shared.tools.gdrive.permissions")


class GoogleDriveCreatePermissionTool(GoogleAPIClient):
    """Share a Google Drive file by creating a permission."""

    name = "google_drive_create_permission"
    description = "Share a Google Drive file by creating a permission (grant access to user/group/domain)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "File ID to share"},
                "role": {
                    "type": "string",
                    "description": "Permission role",
                    "enum": ["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"],
                    "default": "reader"
                },
                "type": {
                    "type": "string",
                    "description": "Permission type",
                    "enum": ["user", "group", "domain", "anyone"],
                    "default": "user"
                },
                "email_address": {"type": "string", "description": "Email for user/group type"},
                "domain": {"type": "string", "description": "Domain for domain type"},
                "allow_file_discovery": {"type": "boolean", "description": "Allow file discovery for domain/anyone"},
                "send_notification_email": {"type": "boolean", "default": True},
                "email_message": {"type": "string"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id", "role", "type"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_permission_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        role = arguments.get("role", "reader")
        perm_type = arguments.get("type", "user")

        permission = {"role": role, "type": perm_type}

        if perm_type in ("user", "group"):
            email = arguments.get("email_address")
            if not email:
                raise HTTPException(status_code=400, detail="email_address required for user/group")
            permission["emailAddress"] = email
        elif perm_type == "domain":
            domain = arguments.get("domain")
            if not domain:
                raise HTTPException(status_code=400, detail="domain required for domain type")
            permission["domain"] = domain

        if arguments.get("allow_file_discovery") is not None:
            permission["allowFileDiscovery"] = arguments["allow_file_discovery"]

        params = {
            "supportsAllDrives": arguments.get("supports_all_drives", True),
            "sendNotificationEmail": arguments.get("send_notification_email", True),
        }

        if arguments.get("email_message"):
            params["emailMessage"] = arguments["email_message"]

        resp = await self._make_request(
            "POST",
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            access_token,
            params=params,
            json_body=permission
        )
        return resp.json()


class GoogleDriveAboutGetTool(GoogleAPIClient):
    """Get Google Drive account information and storage quota."""

    name = "google_drive_about_get"
    description = "Get information about the user's Google Drive account (storage quota, capabilities)."
    required_scopes = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "string",
                    "description": "Fields to retrieve",
                    "default": "user,storageQuota,importFormats,exportFormats,maxUploadSize,canCreateDrives"
                }
            },
            "required": []
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_about_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        params = {
            "fields": arguments.get("fields", "user,storageQuota,importFormats,exportFormats,maxUploadSize,canCreateDrives")
        }

        resp = await self._make_request(
            "GET",
            "https://www.googleapis.com/drive/v3/about",
            access_token,
            params=params
        )
        return resp.json()
