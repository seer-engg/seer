"""
Google Drive write tools: upload, create folder, update, delete, create permission
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import base64
from fastapi import HTTPException

from shared.tools.base import BaseTool
from shared.logger import get_logger
from shared.tools.google._common import (
    _require_access_token,
    _encode_multipart_related,
    _drive_request,
    _drive_file_schema,
    _empty_object_schema,
    _drive_permission_schema,
)

logger = get_logger("shared.tools.google_drive.write")


class GoogleDriveUploadFileTool(BaseTool):
    name = "google_drive_upload_file"
    description = "Upload a file to Google Drive (multipart or media). Expects base64 content."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mime_type": {"type": "string", "default": "application/octet-stream"},
                "parents": {"type": "array", "items": {"type": "string"}},
                "content_base64": {"type": "string"},
                "upload_type": {"type": "string", "enum": ["multipart", "media"], "default": "multipart"},
                "fields": {"type": "string", "default": "id,name,mimeType,parents,modifiedTime,size,webViewLink"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["name", "content_base64"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    # pylint: disable=too-complex,too-many-locals,line-too-long
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive upload tool")
        name = arguments.get("name")
        content_b64 = arguments.get("content_base64")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")
        if not content_b64:
            raise HTTPException(status_code=400, detail="content_base64 is required")

        try:
            content_bytes = base64.b64decode(content_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="content_base64 is not valid base64") from exc

        mime_type = arguments.get("mime_type", "application/octet-stream")
        parents = arguments.get("parents")
        upload_type = arguments.get("upload_type", "multipart")
        fields = arguments.get("fields", "id,name,mimeType,parents,modifiedTime,size,webViewLink")

        headers = {"Authorization": f"Bearer {access_token}"}
        base_url = "https://www.googleapis.com/upload/drive/v3/files"
        params: Dict[str, Any] = {"uploadType": upload_type, "fields": fields, "supportsAllDrives": arguments.get("supports_all_drives", True)}

        metadata: Dict[str, Any] = {"name": name}
        if parents:
            metadata["parents"] = parents
        if mime_type:
            metadata["mimeType"] = mime_type

        if upload_type == "media":
            logger.info("Uploading Drive file (media) name=%s", name)
            headers_post = {**headers, "Content-Type": mime_type}
            content_payload = content_bytes
        else:
            logger.info("Uploading Drive file (multipart) name=%s", name)
            mp = _encode_multipart_related(metadata, content_bytes, mime_type)
            headers_post = {**headers, "Content-Type": mp["content_type"]}
            content_payload = mp["body"]

        try:
            resp = await _drive_request("post", base_url, headers=headers_post, params=params, content=content_payload, timeout=60.0, prefix="Google Drive upload")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected upload error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error uploading Drive file: {str(exc)}") from exc


class GoogleDriveCreateFolderTool(BaseTool):
    name = "google_drive_create_folder"
    description = "Create a folder in Google Drive."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "parents": {"type": "array", "items": {"type": "string"}},
                "fields": {"type": "string", "default": "id,name,mimeType,parents,modifiedTime,webViewLink"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["name"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    # pylint: disable=too-complex,too-many-locals,line-too-long
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive create folder tool")
        name = arguments.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        url = "https://www.googleapis.com/drive/v3/files"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
        if arguments.get("parents"):
            body["parents"] = arguments["parents"]
        params = {"fields": arguments.get("fields", "id,name,mimeType,parents,modifiedTime,webViewLink"), "supportsAllDrives": arguments.get("supports_all_drives", True)}

        try:
            logger.info("Creating Drive folder name=%s", name)
            resp = await _drive_request("post", url, headers=headers, params=params, json_body=body, timeout=30.0, prefix="Google Drive create folder")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected create folder error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error creating Drive folder: {str(exc)}") from exc


class GoogleDriveUpdateFileTool(BaseTool):
    name = "google_drive_update_file"
    description = "Update a Drive file (rename/move; optional content update)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
            "add_parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
            "remove_parents": {
                "resource_type": "google_drive_folder",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            },
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "name": {"type": "string"},
                "add_parents": {"type": "array", "items": {"type": "string"}},
                "remove_parents": {"type": "array", "items": {"type": "string"}},
                "content_base64": {"type": "string"},
                "mime_type": {"type": "string", "default": "application/octet-stream"},
                "fields": {"type": "string", "default": "id,name,mimeType,parents,modifiedTime,size,webViewLink"},
                "supports_all_drives": {"type": "boolean", "default": True},
            },
            "required": ["file_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_file_schema()

    # pylint: disable=too-complex,too-many-locals,line-too-long
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive update tool")
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        headers = {"Authorization": f"Bearer {access_token}"}
        fields = arguments.get("fields", "id,name,mimeType,parents,modifiedTime,size,webViewLink")
        supports_all_drives = arguments.get("supports_all_drives", True)

        add_parents = arguments.get("add_parents") or []
        remove_parents = arguments.get("remove_parents") or []
        add_parents_str = ",".join(add_parents) if add_parents else None
        remove_parents_str = ",".join(remove_parents) if remove_parents else None

        body: Dict[str, Any] = {}
        if arguments.get("name"):
            body["name"] = arguments["name"]

        content_b64 = arguments.get("content_base64")
        do_content_update = bool(content_b64)

        try:
            if do_content_update:
                try:
                    content_bytes = base64.b64decode(content_b64)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail="content_base64 is not valid base64") from exc

                mime_type = arguments.get("mime_type", "application/octet-stream")
                upload_url = f"https://www.googleapis.com/upload/drive/v3/files/{file_id}"
                params: Dict[str, Any] = {"uploadType": "multipart", "fields": fields, "supportsAllDrives": supports_all_drives}
                if add_parents_str:
                    params["addParents"] = add_parents_str
                if remove_parents_str:
                    params["removeParents"] = remove_parents_str

                mp = _encode_multipart_related(body if body else {}, content_bytes, mime_type)
                logger.info("Updating Drive file content file_id=%s", file_id)
                resp = await _drive_request("patch", upload_url, headers={**headers, "Content-Type": mp["content_type"]}, params=params, content=mp["body"], timeout=60.0, prefix="Google Drive files.update (content)")
            else:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
                params = {"fields": fields, "supportsAllDrives": supports_all_drives}
                if add_parents_str:
                    params["addParents"] = add_parents_str
                if remove_parents_str:
                    params["removeParents"] = remove_parents_str

                logger.info("Updating Drive file metadata file_id=%s", file_id)
                resp = await _drive_request("patch", url, headers={**headers, "Content-Type": "application/json"}, params=params, json_body=body, timeout=60.0, prefix="Google Drive files.update (metadata)")

            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected update error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error updating Drive file: {str(exc)}") from exc


class GoogleDriveDeleteFileTool(BaseTool):
    name = "google_drive_delete_file"
    description = "Permanently delete a Drive file by file_id."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"file_id": {"type": "string"}, "supports_all_drives": {"type": "boolean", "default": True}}, "required": ["file_id"]}

    def get_output_schema(self) -> Dict[str, Any]:
        return _empty_object_schema("Drive files.delete success response is an empty JSON object.")

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive delete tool")
        file_id = arguments.get("file_id")
        if not file_id:
            raise HTTPException(status_code=400, detail="file_id is required")

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"supportsAllDrives": arguments.get("supports_all_drives", True)}

        try:
            logger.info("Deleting Drive file file_id=%s", file_id)
            resp = await _drive_request("delete", url, headers=headers, params=params, timeout=30.0, prefix="Google Drive files.delete")
            return resp.json() if resp.text else {}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected delete error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error deleting Drive file: {str(exc)}") from exc


class GoogleDriveCreatePermissionTool(BaseTool):
    name = "google_drive_create_permission"
    description = "Create a sharing permission for a Drive file (share with user/group/domain/anyone)."
    required_scopes = ["https://www.googleapis.com/auth/drive.file"]
    integration_type = "google_drive"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "file_id": {
                "resource_type": "google_drive_file",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "type": {"type": "string", "enum": ["user", "group", "domain", "anyone"]},
                "role": {"type": "string", "enum": ["owner", "organizer", "fileOrganizer", "writer", "commenter", "reader"]},
                "email_address": {"type": "string"},
                "domain": {"type": "string"},
                "allow_file_discovery": {"type": "boolean", "default": False},
                "send_notification_email": {"type": "boolean", "default": True},
                "email_message": {"type": "string"},
                "transfer_ownership": {"type": "boolean", "default": False},
                "move_to_new_owners_root": {"type": "boolean", "default": False},
                "supports_all_drives": {"type": "boolean", "default": True},
                "fields": {"type": "string", "default": "id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime,deleted"},
            },
            "required": ["file_id", "type", "role"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _drive_permission_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token, "Google Drive permission create tool")
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

        url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {"type": p_type, "role": role}
        if arguments.get("email_address"):
            body["emailAddress"] = arguments["email_address"]
        if arguments.get("domain"):
            body["domain"] = arguments["domain"]
        if p_type == "anyone":
            body["allowFileDiscovery"] = arguments.get("allow_file_discovery", False)

        params: Dict[str, Any] = {
            "fields": arguments.get("fields", "id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime,deleted"),
            "supportsAllDrives": arguments.get("supports_all_drives", True),
            "sendNotificationEmail": arguments.get("send_notification_email", True),
            "transferOwnership": arguments.get("transfer_ownership", False),
            "moveToNewOwnersRoot": arguments.get("move_to_new_owners_root", False),
        }
        if arguments.get("email_message"):
            params["emailMessage"] = arguments["email_message"]

        try:
            logger.info("Creating permission file_id=%s type=%s role=%s", file_id, p_type, role)
            resp = await _drive_request("post", url, headers=headers, params=params, json_body=body, timeout=30.0, prefix="Google Drive permissions.create")
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected permission create error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error creating permission: {str(exc)}") from exc
