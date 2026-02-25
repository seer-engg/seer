# pylint: disable=duplicate-code  # Reason: Write tools share picker and schema definitions with other Google tools
"""
Google Docs write operations - creating and updating documents.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gdocs.helpers import (
    _document_output_schema,
    _batch_update_response_schema,
)

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("shared.tools.gdocs.write")


class GoogleDocsWriteTool(GoogleAPIClient):
    """Write/update content in a Google Doc using batchUpdate."""

    name = "google_docs_write"
    description = "Write or update content in a Google Doc. Supports inserting text, deleting content, and replacing text."
    required_scopes = ["https://www.googleapis.com/auth/documents"]
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
                    "description": "The ID of the Google Doc to update",
                },
                "requests": {
                    "type": "array",
                    "description": "Array of update requests. Common types: insertText, deleteContentRange, replaceAllText",
                    "items": {
                        "type": "object",
                        "description": "A single update request",
                    },
                },
            },
            "required": ["document_id", "requests"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _batch_update_response_schema()

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = credentials, context  # unused but required for interface consistency
        document_id = arguments.get("document_id")
        requests = arguments.get("requests")

        if not document_id:
            raise HTTPException(status_code=400, detail="document_id is required")

        if not requests:
            raise HTTPException(status_code=400, detail="requests is required")

        body = {"requests": requests}

        logger.info("Updating Google Doc %s with %d requests", document_id, len(requests))

        resp = await self._make_request(
            "POST",
            f"https://docs.googleapis.com/v1/documents/{document_id}:batchUpdate",
            access_token,
            json_body=body,
        )
        return resp.json()


class GoogleDocsCreateTool(GoogleAPIClient):
    """Create a new Google Doc."""

    name = "google_docs_create"
    description = "Create a new Google Doc with the specified title."
    required_scopes = ["https://www.googleapis.com/auth/documents"]
    integration_type = "google_docs"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the new document",
                },
            },
            "required": ["title"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _document_output_schema()

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = credentials, context  # unused but required for interface consistency
        title = arguments.get("title")

        if not title:
            raise HTTPException(status_code=400, detail="title is required")

        body = {"title": title}

        logger.info("Creating new Google Doc with title: %s", title)

        resp = await self._make_request(
            "POST",
            "https://docs.googleapis.com/v1/documents",
            access_token,
            json_body=body,
        )
        return resp.json()
