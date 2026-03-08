"""
Notion page tools.

Provides tools for reading, creating, updating pages and their content (blocks).
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.notion.base import NotionAPIClient
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.notion.pages")


def _extract_page_title(page: Dict[str, Any]) -> str:
    """Extract title string from a Notion page properties dict."""
    properties = page.get("properties", {})
    for prop_value in properties.values():
        if prop_value.get("type") == "title":
            title_list = prop_value.get("title", [])
            return "".join(part.get("plain_text", "") for part in title_list)
    return "(Untitled)"


def _extract_block_text(block: Dict[str, Any]) -> str:
    """Extract plain text from a Notion block's rich_text array."""
    block_type = block.get("type", "")
    block_data = block.get(block_type, {})
    rich_text = block_data.get("rich_text", [])
    return "".join(part.get("plain_text", "") for part in rich_text)


def _build_paragraph_block(text: str) -> Dict[str, Any]:
    """Build a Notion paragraph block from plain text."""
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {"content": text},
                }
            ]
        },
    }


class NotionGetPageTool(NotionAPIClient):
    """Get a Notion page's properties and metadata by ID."""

    name = "notion_get_page"
    description = (
        "Get a Notion page's properties, title, and metadata. "
        "Returns page properties but not the page body content (use notion_get_page_content for that)."
    )
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID (UUID format, e.g., '8f3b2a1c-...' or without hyphens)",
                },
            },
            "required": ["page_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Page ID"},
                "title": {"type": "string", "description": "Page title"},
                "url": {"type": "string", "description": "URL to open in Notion"},
                "created_time": {"type": "string", "description": "ISO 8601 creation timestamp"},
                "last_edited_time": {"type": "string", "description": "ISO 8601 last edit timestamp"},
                "archived": {"type": "boolean", "description": "Whether the page is archived"},
                "properties": {"type": "object", "description": "Page property values"},
            },
            "required": ["id", "title"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        page_id = arguments.get("page_id")
        if not page_id:
            raise HTTPException(status_code=400, detail="Parameter 'page_id' is required")

        logger.info("Getting Notion page: id=%s", page_id)
        page = await self._make_request("GET", f"pages/{page_id}", credentials=credentials)

        return {
            "id": page.get("id", ""),
            "title": _extract_page_title(page),
            "url": page.get("url", ""),
            "created_time": page.get("created_time", ""),
            "last_edited_time": page.get("last_edited_time", ""),
            "archived": page.get("archived", False),
            "properties": page.get("properties", {}),
        }


class NotionCreatePageTool(NotionAPIClient):
    """Create a new Notion page inside another page or database."""

    name = "notion_create_page"
    description = "Create a new Notion page. Can be created inside an existing page or as a row in a database."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "parent_id": {
                    "type": "string",
                    "description": "ID of the parent page or database",
                },
                "parent_type": {
                    "type": "string",
                    "enum": ["page", "database"],
                    "description": "Whether the parent is a page or a database",
                },
                "title": {
                    "type": "string",
                    "description": "Title of the new page",
                },
                "content": {
                    "type": "string",
                    "description": "Optional initial text content for the page body (added as a paragraph block)",
                },
            },
            "required": ["parent_id", "parent_type", "title"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Created page ID"},
                "url": {"type": "string", "description": "URL to open in Notion"},
            },
            "required": ["id", "url"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        parent_id = arguments.get("parent_id")
        parent_type = arguments.get("parent_type")
        title = arguments.get("title")

        if not parent_id:
            raise HTTPException(status_code=400, detail="Parameter 'parent_id' is required")
        if not parent_type:
            raise HTTPException(status_code=400, detail="Parameter 'parent_type' is required")
        if not title:
            raise HTTPException(status_code=400, detail="Parameter 'title' is required")

        # Build parent reference
        if parent_type == "database":
            parent = {"database_id": parent_id}
        else:
            parent = {"page_id": parent_id}

        # Build title property
        body: Dict[str, Any] = {
            "parent": parent,
            "properties": {
                "title": {
                    "title": [{"type": "text", "text": {"content": title}}]
                }
            },
        }

        # Optionally add content as a paragraph block
        if arguments.get("content"):
            body["children"] = [_build_paragraph_block(arguments["content"])]

        logger.info("Creating Notion page: title=%s, parent_type=%s, parent_id=%s", title, parent_type, parent_id)
        page = await self._make_request("POST", "pages", credentials=credentials, json_body=body)

        return {
            "id": page.get("id", ""),
            "url": page.get("url", ""),
        }


class NotionUpdatePageTool(NotionAPIClient):
    """Update a Notion page's title or archive status."""

    name = "notion_update_page"
    description = "Update a Notion page's title or archive/unarchive it."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID to update",
                },
                "title": {
                    "type": "string",
                    "description": "New title for the page (optional)",
                },
                "archived": {
                    "type": "boolean",
                    "description": "Set to true to archive the page, false to unarchive (optional)",
                },
            },
            "required": ["page_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Updated page ID"},
                "url": {"type": "string", "description": "URL to open in Notion"},
            },
            "required": ["id", "url"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        page_id = arguments.get("page_id")
        if not page_id:
            raise HTTPException(status_code=400, detail="Parameter 'page_id' is required")

        body: Dict[str, Any] = {}

        if arguments.get("title") is not None:
            body["properties"] = {
                "title": {
                    "title": [{"type": "text", "text": {"content": arguments["title"]}}]
                }
            }

        if arguments.get("archived") is not None:
            body["archived"] = arguments["archived"]

        if not body:
            raise HTTPException(status_code=400, detail="At least one of 'title' or 'archived' must be provided")

        logger.info("Updating Notion page: id=%s", page_id)
        page = await self._make_request("PATCH", f"pages/{page_id}", credentials=credentials, json_body=body)

        return {
            "id": page.get("id", ""),
            "url": page.get("url", ""),
        }


class NotionGetPageContentTool(NotionAPIClient):
    """Get the content blocks of a Notion page."""

    name = "notion_get_page_content"
    description = "Get the text content (blocks) of a Notion page. Returns the page body as a list of block objects with their type and text."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID to read content from",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Maximum number of blocks to return (default: 100, max: 100)",
                    "default": 100,
                    "maximum": 100,
                },
            },
            "required": ["page_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "description": "List of content blocks",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Block ID"},
                            "type": {"type": "string", "description": "Block type (paragraph, heading_1, etc.)"},
                            "text": {"type": "string", "description": "Plain text content of the block"},
                            "has_children": {"type": "boolean", "description": "Whether this block has nested children"},
                        },
                    },
                },
                "has_more": {"type": "boolean", "description": "Whether more blocks are available"},
            },
            "required": ["blocks"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        page_id = arguments.get("page_id")
        if not page_id:
            raise HTTPException(status_code=400, detail="Parameter 'page_id' is required")

        params: Dict[str, Any] = {
            "page_size": min(arguments.get("page_size", 100), 100),
        }

        logger.info("Getting Notion page content: page_id=%s", page_id)
        response = await self._make_request(
            "GET", f"blocks/{page_id}/children", credentials=credentials, params=params
        )

        raw_blocks: List[Dict[str, Any]] = response.get("results", [])
        blocks = [
            {
                "id": b.get("id", ""),
                "type": b.get("type", ""),
                "text": _extract_block_text(b),
                "has_children": b.get("has_children", False),
            }
            for b in raw_blocks
        ]

        logger.info("Retrieved %d blocks from page %s", len(blocks), page_id)
        return {"blocks": blocks, "has_more": response.get("has_more", False)}


class NotionAppendPageContentTool(NotionAPIClient):
    """Append text content to a Notion page."""

    name = "notion_append_page_content"
    description = "Append text to a Notion page as a new paragraph block."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "page_id": {
                    "type": "string",
                    "description": "Notion page ID to append content to",
                },
                "text": {
                    "type": "string",
                    "description": "Text to append as a paragraph block",
                },
            },
            "required": ["page_id", "text"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "block_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of the newly created blocks",
                },
            },
            "required": ["block_ids"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context

        page_id = arguments.get("page_id")
        text = arguments.get("text")

        if not page_id:
            raise HTTPException(status_code=400, detail="Parameter 'page_id' is required")
        if not text:
            raise HTTPException(status_code=400, detail="Parameter 'text' is required")

        body = {"children": [_build_paragraph_block(text)]}

        logger.info("Appending content to Notion page: page_id=%s", page_id)
        response = await self._make_request(
            "PATCH", f"blocks/{page_id}/children", credentials=credentials, json_body=body
        )

        created_blocks: List[Dict[str, Any]] = response.get("results", [])
        block_ids = [b.get("id", "") for b in created_blocks]

        logger.info("Appended %d blocks to page %s", len(block_ids), page_id)
        return {"block_ids": block_ids}
