# pylint: disable=duplicate-code  # Reason: Read tools share picker and schema definitions with other Google tools
"""
Google Docs read operations - reading document content.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gdocs.helpers import _document_output_schema

logger = get_logger("shared.tools.gdocs.read")


class GoogleDocsReadTool(GoogleAPIClient):
    """Read content from a Google Doc."""

    name = "google_docs_read"
    description = "Read content from a Google Doc."
    required_scopes = ["https://www.googleapis.com/auth/documents.readonly"]
    integration_type = "google_docs"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "document_id": {
                "resource_type": "google_document",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "hierarchy": False,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The ID of the Google Doc to read",
                },
                "suggestions_view_mode": {
                    "type": "string",
                    "enum": [
                        "DEFAULT_FOR_CURRENT_ACCESS",
                        "SUGGESTIONS_INLINE",
                        "PREVIEW_SUGGESTIONS_ACCEPTED",
                        "PREVIEW_WITHOUT_SUGGESTIONS",
                    ],
                    "default": "DEFAULT_FOR_CURRENT_ACCESS",
                    "description": "The suggestions view mode to apply to the document",
                },
            },
            "required": ["document_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _document_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        document_id = arguments.get("document_id")

        if not document_id:
            raise HTTPException(status_code=400, detail="document_id is required")

        params = {}
        suggestions_view_mode = arguments.get("suggestions_view_mode")
        if suggestions_view_mode:
            params["suggestionsViewMode"] = suggestions_view_mode

        logger.info("Reading Google Doc %s", document_id)

        resp = await self._make_request(
            "GET",
            f"https://docs.googleapis.com/v1/documents/{document_id}",
            access_token,
            params=params if params else None,
        )
        return resp.json()
