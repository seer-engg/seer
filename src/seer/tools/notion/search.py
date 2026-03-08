"""
Notion search tool.

Provides full-text search across all pages and databases accessible
to the connected Notion integration.
"""
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.notion.base import NotionAPIClient
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.notion.search")


def _extract_title(result: Dict[str, Any]) -> str:
    """Extract title from a Notion search result (page or database)."""
    object_type = result.get("object", "")
    properties = result.get("properties", {})

    if object_type == "database":
        # Database title is in result["title"] list
        title_parts = result.get("title", [])
        return "".join(part.get("plain_text", "") for part in title_parts)

    # Page - find the title property
    for prop_value in properties.values():
        if prop_value.get("type") == "title":
            title_list = prop_value.get("title", [])
            return "".join(part.get("plain_text", "") for part in title_list)

    return "(Untitled)"


class NotionSearchTool(NotionAPIClient):
    """Search across all pages and databases in the connected Notion workspace."""

    name = "notion_search"
    description = "Search Notion for pages and databases by query text. Returns matching results with their IDs and URLs."
    required_scopes = []
    integration_type = "notion"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text",
                },
                "filter_type": {
                    "type": "string",
                    "enum": ["page", "database"],
                    "description": "Filter results to only pages or only databases (optional, returns both by default)",
                },
                "sort_direction": {
                    "type": "string",
                    "enum": ["ascending", "descending"],
                    "description": "Sort direction for results by last edited time (optional)",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 100)",
                    "default": 10,
                    "maximum": 100,
                },
            },
            "required": ["query"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "Matching pages and databases",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Page or database ID"},
                            "title": {"type": "string", "description": "Page or database title"},
                            "type": {"type": "string", "description": "Result type: 'page' or 'database'"},
                            "url": {"type": "string", "description": "URL to open in Notion"},
                            "last_edited_time": {"type": "string", "description": "ISO 8601 timestamp of last edit"},
                        },
                    },
                },
                "has_more": {
                    "type": "boolean",
                    "description": "Whether more results are available",
                },
            },
            "required": ["results"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context  # Unused but required for interface consistency

        query = arguments.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Parameter 'query' is required")

        body: Dict[str, Any] = {
            "query": query,
            "page_size": min(arguments.get("page_size", 10), 100),
        }

        if arguments.get("filter_type"):
            body["filter"] = {"value": arguments["filter_type"], "property": "object"}

        if arguments.get("sort_direction"):
            body["sort"] = {
                "direction": arguments["sort_direction"],
                "timestamp": "last_edited_time",
            }

        logger.info("Searching Notion: query=%s, filter=%s", query, arguments.get("filter_type"))

        response = await self._make_request("POST", "search", credentials=credentials, json_body=body)

        raw_results: List[Dict[str, Any]] = response.get("results", [])
        results = [
            {
                "id": r.get("id", ""),
                "title": _extract_title(r),
                "type": r.get("object", ""),
                "url": r.get("url", ""),
                "last_edited_time": r.get("last_edited_time", ""),
            }
            for r in raw_results
        ]

        logger.info("Notion search returned %d results for query='%s'", len(results), query)
        return {"results": results, "has_more": response.get("has_more", False)}
