from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.services.integrations.resource_providers.base import ResourceProvider
from seer.logger import get_logger

logger = get_logger(__name__)


class GoogleResourceProvider(ResourceProvider):
    provider = "google"
    aliases = {"gmail", "googlesheets", "googledrive"}
    resource_configs: Dict[str, Dict[str, Any]] = {
        "google_drive_file": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,iconLink,webViewLink)",
            "supports_hierarchy": True,
            "supports_search": True,
        },
        "google_spreadsheet": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,modifiedTime,iconLink,webViewLink)",
            "default_query": "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
            "supports_hierarchy": False,
            "supports_search": True,
        },
        "google_sheet_tab": {
            "list_endpoint": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            "display_field": "title",
            "value_field": "title",
            "depends_on": "spreadsheet_id",
            "supports_hierarchy": False,
            "supports_search": False,
        },
        "google_drive_folder": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,iconLink,webViewLink)",
            "default_query": "mimeType='application/vnd.google-apps.folder' and trashed=false",
            "supports_hierarchy": True,
            "supports_search": True,
        },
        "gmail_label": {
            "list_endpoint": "https://gmail.googleapis.com/gmail/v1/users/me/labels",
            "display_field": "name",
            "value_field": "id",
            "supports_hierarchy": False,
            "supports_search": False,
        },
    }

    async def list_resources(
        self,
        *,
        access_token: str,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        config = self.resource_configs.get(resource_type)
        if not config:
            raise HTTPException(status_code=400, detail=f"Unsupported Google resource type '{resource_type}'")

        if resource_type == "google_spreadsheet":
            logger.info(f"Listing Google Spreadsheets for parent_id: {parent_id}")
            return await self._list_drive_files(
                access_token,
                config,
                query=query,
                parent_id=parent_id,
                page_token=page_token,
                page_size=page_size,
                mime_type="application/vnd.google-apps.spreadsheet",
            )
        if resource_type == "google_drive_file":
            mime_type = (filter_params or {}).get("mimeType")
            return await self._list_drive_files(
                access_token,
                config,
                query=query,
                parent_id=parent_id,
                page_token=page_token,
                page_size=page_size,
                mime_type=mime_type,
            )
        if resource_type == "google_drive_folder":
            return await self._list_drive_files(
                access_token,
                config,
                query=query,
                parent_id=parent_id,
                page_token=page_token,
                page_size=page_size,
                mime_type="application/vnd.google-apps.folder",
            )
        if resource_type == "google_sheet_tab":
            spreadsheet_id = (depends_on_values or {}).get("spreadsheet_id")
            if not spreadsheet_id:
                return {"items": [], "error": "spreadsheet_id is required"}
            return await self._list_google_sheet_tabs(access_token, spreadsheet_id)
        if resource_type == "gmail_label":
            return await self._list_gmail_labels(access_token)

        raise HTTPException(status_code=400, detail=f"Unhandled Google resource type '{resource_type}'")

    async def _list_drive_files(
        self,
        access_token: str,
        config: Dict[str, Any],
        *,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        mime_type: Optional[str],
    ) -> Dict[str, Any]:
        url = config["list_endpoint"]
        headers = {"Authorization": f"Bearer {access_token}"}

        q_parts = ["trashed=false"]
        if mime_type:
            q_parts.append(f"mimeType='{mime_type}'")
        if parent_id:
            q_parts.append(f"'{parent_id}' in parents")
        if query:
            q_parts.append(f"name contains '{query}'")

        params = {
            "q": " and ".join(q_parts),
            "pageSize": page_size,
            "fields": config.get("default_fields", "nextPageToken,files(id,name,mimeType)"),
            "orderBy": "folder,name",
            "supportsAllDrives": True,
            "includeItemsFromAllDrives": True,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            logger.exception("Error listing Google Drive files: %s", exc)
            return {"items": [], "error": str(exc), "next_page_token": None}

        if response.status_code != 200:
            logger.error("Google Drive API error: %s - %s", response.status_code, response.text[:200])
            return {"items": [], "error": f"API error: {response.status_code}", "next_page_token": None}

        data = response.json()
        items = []
        for file_data in data.get("files", []):
            is_folder = file_data.get("mimeType") == "application/vnd.google-apps.folder"
            items.append(
                {
                    "id": file_data.get("id"),
                    "name": file_data.get("name"),
                    "display_name": file_data.get("name"),
                    "type": "folder" if is_folder else "file",
                    "mime_type": file_data.get("mimeType"),
                    "icon_url": file_data.get("iconLink"),
                    "web_url": file_data.get("webViewLink"),
                    "modified_time": file_data.get("modifiedTime"),
                    "has_children": is_folder,
                }
            )

        return {
            "items": items,
            "next_page_token": data.get("nextPageToken"),
            "total_count": len(items),
            "supports_hierarchy": config.get("supports_hierarchy", False),
            "supports_search": config.get("supports_search", True),
        }

    async def _list_google_sheet_tabs(self, access_token: str, spreadsheet_id: str) -> Dict[str, Any]:
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
        headers = {"Authorization": f"Bearer {access_token}"}
        params = {"fields": "sheets.properties"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
        except Exception as exc:
            logger.exception("Error listing Google Sheet tabs: %s", exc)
            return {"items": [], "error": str(exc)}

        if response.status_code != 200:
            logger.error("Google Sheets API error: %s - %s", response.status_code, response.text[:200])
            return {"items": [], "error": f"API error: {response.status_code}"}

        data = response.json()
        items = []
        for sheet in data.get("sheets", []):
            props = sheet.get("properties", {})
            items.append(
                {
                    "id": str(props.get("sheetId")),
                    "name": props.get("title"),
                    "display_name": props.get("title"),
                    "type": "sheet_tab",
                    "index": props.get("index"),
                }
            )

        return {"items": items, "supports_hierarchy": False, "supports_search": False}

    async def _list_gmail_labels(self, access_token: str) -> Dict[str, Any]:
        url = "https://gmail.googleapis.com/gmail/v1/users/me/labels"
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
        except Exception as exc:
            logger.exception("Error listing Gmail labels: %s", exc)
            return {"items": [], "error": str(exc)}

        if response.status_code != 200:
            logger.error("Gmail API error: %s - %s", response.status_code, response.text[:200])
            return {"items": [], "error": f"API error: {response.status_code}"}

        data = response.json()
        items = []
        for label in data.get("labels", []):
            items.append(
                {
                    "id": label.get("id"),
                    "name": label.get("name"),
                    "display_name": label.get("name"),
                    "type": "label",
                    "label_type": label.get("type"),
                }
            )

        return {"items": items, "supports_hierarchy": False, "supports_search": False}
