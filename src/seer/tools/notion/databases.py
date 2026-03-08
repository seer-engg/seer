"""
Notion database tools.

Provides tools for querying databases and creating database entries (rows).
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.notion.base import NotionAPIClient
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.notion.databases")


def _extract_entry_title(page: Dict[str, Any]) -> str:
    """Extract title from a database entry (page with database parent)."""
    properties = page.get("properties", {})
    for prop_value in properties.values():
        if prop_value.get("type") == "title":
            title_list = prop_value.get("title", [])
            return "".join(part.get("plain_text", "") for part in title_list)
    return "(Untitled)"


_RICH_TEXT_TYPES = {"title", "rich_text"}
_SCALAR_TYPES = {"number", "url", "email", "phone_number", "checkbox"}
_NAMED_OBJECT_TYPES = {"select", "status"}


def _format_property_value(prop: Dict[str, Any]) -> Any:  # pylint: disable=too-many-return-statements
    # Reason: dispatch table reduces McCabe complexity; return statements map 1-to-1 to distinct Notion property types
    """Extract a human-readable value from a Notion property."""
    prop_type = prop.get("type", "")

    if prop_type in _RICH_TEXT_TYPES:
        return "".join(p.get("plain_text", "") for p in prop.get(prop_type, []))
    if prop_type in _SCALAR_TYPES:
        return prop.get(prop_type)
    if prop_type in _NAMED_OBJECT_TYPES:
        return (prop.get(prop_type) or {}).get("name")
    if prop_type == "multi_select":
        return [opt.get("name") for opt in prop.get("multi_select", [])]
    if prop_type == "date":
        return (prop.get("date") or {}).get("start")

    # Return raw value for unsupported types
    return prop.get(prop_type)


class NotionQueryDatabaseTool(NotionAPIClient):
    """Query a Notion database with optional filters and sorts."""

    name = "notion_query_database"
    description = "Query a Notion database and return its entries (rows). Supports filtering and sorting."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "string",
                    "description": "Notion database ID (UUID format)",
                },
                "filter": {
                    "type": "object",
                    "description": "Notion filter object (optional). See Notion API docs for filter syntax.",
                    "additionalProperties": True,
                },
                "sorts": {
                    "type": "array",
                    "description": "Array of sort objects (optional). Each has 'property' (string) and 'direction' ('ascending'|'descending').",
                    "items": {
                        "type": "object",
                        "properties": {
                            "property": {"type": "string", "description": "Property name to sort by"},
                            "direction": {
                                "type": "string",
                                "enum": ["ascending", "descending"],
                                "description": "Sort direction",
                            },
                        },
                        "required": ["property"],
                    },
                },
                "page_size": {
                    "type": "integer",
                    "description": "Maximum number of entries to return (default: 10, max: 100)",
                    "default": 10,
                    "maximum": 100,
                },
            },
            "required": ["database_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "description": "Database entries with their properties",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Entry (page) ID"},
                            "title": {"type": "string", "description": "Entry title"},
                            "url": {"type": "string", "description": "URL to open in Notion"},
                            "created_time": {"type": "string", "description": "Creation timestamp"},
                            "last_edited_time": {"type": "string", "description": "Last edit timestamp"},
                            "properties": {"type": "object", "description": "Property values (simplified)"},
                        },
                    },
                },
                "has_more": {"type": "boolean", "description": "Whether more entries are available"},
            },
            "required": ["entries"],
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

        database_id = arguments.get("database_id")
        if not database_id:
            raise HTTPException(status_code=400, detail="Parameter 'database_id' is required")

        body: Dict[str, Any] = {
            "page_size": min(arguments.get("page_size", 10), 100),
        }

        if arguments.get("filter"):
            body["filter"] = arguments["filter"]

        if arguments.get("sorts"):
            body["sorts"] = arguments["sorts"]

        logger.info("Querying Notion database: id=%s", database_id)
        response = await self._make_request(
            "POST", f"databases/{database_id}/query", credentials=credentials, json_body=body
        )

        raw_results: List[Dict[str, Any]] = response.get("results", [])
        entries = [
            {
                "id": entry.get("id", ""),
                "title": _extract_entry_title(entry),
                "url": entry.get("url", ""),
                "created_time": entry.get("created_time", ""),
                "last_edited_time": entry.get("last_edited_time", ""),
                "properties": {
                    k: _format_property_value(v)
                    for k, v in entry.get("properties", {}).items()
                },
            }
            for entry in raw_results
        ]

        logger.info("Notion database query returned %d entries from database %s", len(entries), database_id)
        return {"entries": entries, "has_more": response.get("has_more", False)}


class NotionCreateDatabaseEntryTool(NotionAPIClient):
    """Create a new entry (row) in a Notion database."""

    name = "notion_create_database_entry"
    description = "Create a new entry (row) in a Notion database with the specified property values."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "string",
                    "description": "Notion database ID to add the entry to",
                },
                "properties": {
                    "type": "object",
                    "description": (
                        "Property values for the new entry, matching the database schema. "
                        "Use Notion property format: e.g., for a title property: "
                        "{\"Name\": {\"title\": [{\"text\": {\"content\": \"My Entry\"}}]}}, "
                        "for a select: {\"Status\": {\"select\": {\"name\": \"In Progress\"}}}"
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["database_id", "properties"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Created entry (page) ID"},
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

        database_id = arguments.get("database_id")
        properties = arguments.get("properties")

        if not database_id:
            raise HTTPException(status_code=400, detail="Parameter 'database_id' is required")
        if not properties:
            raise HTTPException(status_code=400, detail="Parameter 'properties' is required")

        body: Dict[str, Any] = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        logger.info("Creating Notion database entry in database: id=%s", database_id)
        page = await self._make_request("POST", "pages", credentials=credentials, json_body=body)

        return {
            "id": page.get("id", ""),
            "url": page.get("url", ""),
        }
