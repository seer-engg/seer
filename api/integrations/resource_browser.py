"""
Resource Browser for Integration Resources

This module provides a unified interface for browsing resources from
various integrations (Google Drive, Google Sheets, GitHub, etc.).

The ResourceBrowser class handles:
- Listing resources with pagination
- Searching resources
- Navigating folder hierarchies
- Filtering by resource type (e.g., only spreadsheets)
"""
from typing import Any, Dict, List, Optional
import httpx
from shared.logger import get_logger

logger = get_logger("api.integrations.resource_browser")


class ResourceBrowser:
    """
    Unified resource browser for integration resources.
    
    Supports multiple resource types across different providers:
    - google: drive_files, sheets, gmail_labels
    - github: repos, branches, issues
    """
    
    # Resource type configurations
    RESOURCE_CONFIGS = {
        # Google Drive files
        "google_drive_file": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,iconLink,webViewLink)",
            "supports_hierarchy": True,
            "supports_search": True,
        },
        # Google Sheets (specialized filter on drive files)
        "google_spreadsheet": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,modifiedTime,iconLink,webViewLink)",
            "default_query": "mimeType='application/vnd.google-apps.spreadsheet' and trashed=false",
            "supports_hierarchy": False,  # Flat list for sheets
            "supports_search": True,
        },
        # Google Sheets tabs (requires spreadsheet_id)
        "google_sheet_tab": {
            "list_endpoint": "https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            "display_field": "title",
            "value_field": "title",
            "depends_on": "spreadsheet_id",
            "supports_hierarchy": False,
            "supports_search": False,
        },
        # Google Drive folders
        "google_drive_folder": {
            "list_endpoint": "https://www.googleapis.com/drive/v3/files",
            "display_field": "name",
            "value_field": "id",
            "default_fields": "nextPageToken,files(id,name,mimeType,parents,modifiedTime,iconLink,webViewLink)",
            "default_query": "mimeType='application/vnd.google-apps.folder' and trashed=false",
            "supports_hierarchy": True,
            "supports_search": True,
        },
        # Gmail labels
        "gmail_label": {
            "list_endpoint": "https://gmail.googleapis.com/gmail/v1/users/me/labels",
            "display_field": "name",
            "value_field": "id",
            "supports_hierarchy": False,
            "supports_search": False,
        },
        # GitHub repositories
        "github_repo": {
            "list_endpoint": "https://api.github.com/user/repos",
            "display_field": "full_name",
            "value_field": "full_name",
            "supports_hierarchy": False,
            "supports_search": True,
        },
        # GitHub branches (requires repo)
        "github_branch": {
            "list_endpoint": "https://api.github.com/repos/{owner}/{repo}/branches",
            "display_field": "name",
            "value_field": "name",
            "depends_on": "repo",
            "supports_hierarchy": False,
            "supports_search": False,
        },
    }
    
    def __init__(self, access_token: str, provider: str):
        """
        Initialize resource browser.
        
        Args:
            access_token: OAuth access token for API calls
            provider: OAuth provider (google, github, etc.)
        """
        self.access_token = access_token
        self.provider = provider
    
    async def list_resources(
        self,
        resource_type: str,
        query: Optional[str] = None,
        parent_id: Optional[str] = None,
        page_token: Optional[str] = None,
        page_size: int = 50,
        filter_params: Optional[Dict[str, Any]] = None,
        depends_on_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        List resources of a given type.
        
        Args:
            resource_type: Type of resource to list (e.g., 'google_drive_file')
            query: Search query string
            parent_id: Parent folder ID for hierarchical navigation
            page_token: Pagination token
            page_size: Number of results per page
            filter_params: Additional filter parameters
            depends_on_values: Values for dependent parameters (e.g., spreadsheet_id for tabs)
        
        Returns:
            Dict with 'items', 'next_page_token', and metadata
        """
        if resource_type not in self.RESOURCE_CONFIGS:
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        config = self.RESOURCE_CONFIGS[resource_type]
        
        # Handle different resource types
        if resource_type == "google_spreadsheet":
            return await self._list_google_drive_files(
                config, query, parent_id, page_token, page_size,
                mime_type="application/vnd.google-apps.spreadsheet"
            )
        elif resource_type == "google_drive_file":
            mime_type = filter_params.get("mimeType") if filter_params else None
            return await self._list_google_drive_files(
                config, query, parent_id, page_token, page_size, mime_type
            )
        elif resource_type == "google_drive_folder":
            return await self._list_google_drive_files(
                config, query, parent_id, page_token, page_size,
                mime_type="application/vnd.google-apps.folder"
            )
        elif resource_type == "google_sheet_tab":
            spreadsheet_id = depends_on_values.get("spreadsheet_id") if depends_on_values else None
            if not spreadsheet_id:
                return {"items": [], "error": "spreadsheet_id is required"}
            return await self._list_google_sheet_tabs(spreadsheet_id)
        elif resource_type == "gmail_label":
            return await self._list_gmail_labels()
        elif resource_type == "github_repo":
            return await self._list_github_repos(query, page_token, page_size)
        elif resource_type == "github_branch":
            repo = depends_on_values.get("repo") if depends_on_values else None
            if not repo:
                return {"items": [], "error": "repo is required"}
            return await self._list_github_branches(repo)
        else:
            return {"items": [], "error": f"Resource type {resource_type} not implemented"}
    
    async def _list_google_drive_files(
        self,
        config: Dict[str, Any],
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List Google Drive files with optional filters."""
        url = config["list_endpoint"]
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Build query
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
                
                if response.status_code != 200:
                    logger.error(f"Google Drive API error: {response.status_code} - {response.text[:200]}")
                    return {
                        "items": [],
                        "error": f"API error: {response.status_code}",
                        "next_page_token": None,
                    }
                
                data = response.json()
                files = data.get("files", [])
                
                # Transform to standard format
                items = []
                for f in files:
                    is_folder = f.get("mimeType") == "application/vnd.google-apps.folder"
                    items.append({
                        "id": f.get("id"),
                        "name": f.get("name"),
                        "display_name": f.get("name"),
                        "type": "folder" if is_folder else "file",
                        "mime_type": f.get("mimeType"),
                        "icon_url": f.get("iconLink"),
                        "web_url": f.get("webViewLink"),
                        "modified_time": f.get("modifiedTime"),
                        "has_children": is_folder,
                    })
                
                return {
                    "items": items,
                    "next_page_token": data.get("nextPageToken"),
                    "total_count": len(items),
                    "supports_hierarchy": config.get("supports_hierarchy", False),
                    "supports_search": config.get("supports_search", True),
                }
                
        except Exception as e:
            logger.exception(f"Error listing Google Drive files: {e}")
            return {
                "items": [],
                "error": str(e),
                "next_page_token": None,
            }
    
    async def _list_google_sheet_tabs(self, spreadsheet_id: str) -> Dict[str, Any]:
        """List tabs/sheets within a Google Spreadsheet."""
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        params = {"fields": "sheets.properties"}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Google Sheets API error: {response.status_code} - {response.text[:200]}")
                    return {
                        "items": [],
                        "error": f"API error: {response.status_code}",
                    }
                
                data = response.json()
                sheets = data.get("sheets", [])
                
                items = []
                for sheet in sheets:
                    props = sheet.get("properties", {})
                    items.append({
                        "id": str(props.get("sheetId")),
                        "name": props.get("title"),
                        "display_name": props.get("title"),
                        "type": "sheet_tab",
                        "index": props.get("index"),
                    })
                
                return {
                    "items": items,
                    "supports_hierarchy": False,
                    "supports_search": False,
                }
                
        except Exception as e:
            logger.exception(f"Error listing Google Sheet tabs: {e}")
            return {
                "items": [],
                "error": str(e),
            }
    
    async def _list_gmail_labels(self) -> Dict[str, Any]:
        """List Gmail labels."""
        url = "https://gmail.googleapis.com/gmail/v1/users/me/labels"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"Gmail API error: {response.status_code} - {response.text[:200]}")
                    return {
                        "items": [],
                        "error": f"API error: {response.status_code}",
                    }
                
                data = response.json()
                labels = data.get("labels", [])
                
                items = []
                for label in labels:
                    items.append({
                        "id": label.get("id"),
                        "name": label.get("name"),
                        "display_name": label.get("name"),
                        "type": "label",
                        "label_type": label.get("type"),
                    })
                
                return {
                    "items": items,
                    "supports_hierarchy": False,
                    "supports_search": False,
                }
                
        except Exception as e:
            logger.exception(f"Error listing Gmail labels: {e}")
            return {
                "items": [],
                "error": str(e),
            }
    
    async def _list_github_repos(
        self,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List GitHub repositories."""
        url = "https://api.github.com/user/repos"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        params = {
            "per_page": page_size,
            "sort": "updated",
            "direction": "desc",
        }
        
        if page_token:
            params["page"] = page_token
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"GitHub API error: {response.status_code} - {response.text[:200]}")
                    return {
                        "items": [],
                        "error": f"API error: {response.status_code}",
                    }
                
                repos = response.json()
                
                # Filter by query if provided
                if query:
                    query_lower = query.lower()
                    repos = [r for r in repos if query_lower in r.get("full_name", "").lower()]
                
                items = []
                for repo in repos:
                    items.append({
                        "id": str(repo.get("id")),
                        "name": repo.get("full_name"),
                        "display_name": repo.get("full_name"),
                        "type": "repository",
                        "description": repo.get("description"),
                        "private": repo.get("private"),
                        "web_url": repo.get("html_url"),
                    })
                
                # Parse Link header for pagination
                link_header = response.headers.get("Link", "")
                next_page = None
                if 'rel="next"' in link_header:
                    # Extract next page number
                    import re
                    match = re.search(r'page=(\d+)>; rel="next"', link_header)
                    if match:
                        next_page = match.group(1)
                
                return {
                    "items": items,
                    "next_page_token": next_page,
                    "supports_hierarchy": False,
                    "supports_search": True,
                }
                
        except Exception as e:
            logger.exception(f"Error listing GitHub repos: {e}")
            return {
                "items": [],
                "error": str(e),
            }
    
    async def _list_github_branches(self, repo: str) -> Dict[str, Any]:
        """List branches for a GitHub repository."""
        # repo should be in format "owner/repo"
        url = f"https://api.github.com/repos/{repo}/branches"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/vnd.github+json",
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers)
                
                if response.status_code != 200:
                    logger.error(f"GitHub API error: {response.status_code} - {response.text[:200]}")
                    return {
                        "items": [],
                        "error": f"API error: {response.status_code}",
                    }
                
                branches = response.json()
                
                items = []
                for branch in branches:
                    items.append({
                        "id": branch.get("name"),
                        "name": branch.get("name"),
                        "display_name": branch.get("name"),
                        "type": "branch",
                        "protected": branch.get("protected", False),
                    })
                
                return {
                    "items": items,
                    "supports_hierarchy": False,
                    "supports_search": False,
                }
                
        except Exception as e:
            logger.exception(f"Error listing GitHub branches: {e}")
            return {
                "items": [],
                "error": str(e),
            }
    
    @classmethod
    def get_supported_resource_types(cls, provider: str) -> List[str]:
        """Get list of supported resource types for a provider."""
        provider_prefix = {
            "google": ["google_", "gmail_"],
            "github": ["github_"],
        }
        
        prefixes = provider_prefix.get(provider, [])
        return [
            rt for rt in cls.RESOURCE_CONFIGS.keys()
            if any(rt.startswith(p) for p in prefixes)
        ]
    
    @classmethod
    def get_resource_type_info(cls, resource_type: str) -> Optional[Dict[str, Any]]:
        """Get configuration info for a resource type."""
        if resource_type not in cls.RESOURCE_CONFIGS:
            return None
        
        config = cls.RESOURCE_CONFIGS[resource_type]
        return {
            "resource_type": resource_type,
            "display_field": config.get("display_field", "name"),
            "value_field": config.get("value_field", "id"),
            "supports_hierarchy": config.get("supports_hierarchy", False),
            "supports_search": config.get("supports_search", False),
            "depends_on": config.get("depends_on"),
        }

