"""
Google Drive folder operations - create and delete.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gdrive.helpers import (
    _drive_file_schema,
    _empty_object_schema,
)

logger = get_logger("shared.tools.gdrive.folders")


class GoogleDriveCreateFolderTool(GoogleAPIClient):
    """Create a new folder in Google Drive."""

    name = "google_drive_create_folder"
    description = "Create a new folder in Google Drive."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Folder name"},
                "parent_folder_id": {"type": "string", "description": "Parent folder ID"},
                "description": {"type": "string"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["name"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        name = arguments.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        metadata = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder"
        }

        if arguments.get("parent_folder_id"):
            metadata["parents"] = [arguments["parent_folder_id"]]
        if arguments.get("description"):
            metadata["description"] = arguments["description"]

        params = {"supportsAllDrives": arguments.get("supports_all_drives", True)}

        resp = await self._make_request(
            "POST",
            "https://www.googleapis.com/drive/v3/files",
            access_token,
            params=params,
            json_body=metadata
        )
        return resp.json()


class GoogleDriveDeleteFileTool(GoogleAPIClient):
    """Delete a Google Drive file or folder."""

    name = "google_drive_delete_file"
    description = "Delete a Google Drive file or folder (moves to trash by default)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "File/folder ID to delete"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _empty_object_schema("File deleted successfully")

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        params = {"supportsAllDrives": arguments.get("supports_all_drives", True)}

        await self._make_request(
            "DELETE",
            f"https://www.googleapis.com/drive/v3/files/{file_id}",
            access_token,
            params=params
        )
        return {"status": "deleted", "file_id": file_id}
