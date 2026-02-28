"""
Airtable base and table operations.

Provides tools for listing bases and tables with their schemas.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.airtable.base import AirtableAPIClient
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.airtable.bases")


class AirtableListBasesTool(AirtableAPIClient):
    """List all accessible Airtable bases."""

    name = "airtable_list_bases"
    description = "List all Airtable bases accessible to the connected account."
    required_scopes = ["schema.bases:read"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "offset": {
                    "type": "string",
                    "description": "Pagination offset from previous response (optional)",
                },
            },
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bases": {
                    "type": "array",
                    "description": "List of accessible bases",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Base ID"},
                            "name": {"type": "string", "description": "Base name"},
                            "permissionLevel": {"type": "string", "description": "User's permission level"},
                        },
                    },
                },
                "offset": {
                    "type": "string",
                    "description": "Pagination offset for next page (if more results exist)",
                },
            },
            "required": ["bases"],
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

        params: Dict[str, Any] = {}
        offset = arguments.get("offset")
        if offset:
            params["offset"] = offset

        logger.info("Listing Airtable bases")

        response = await self._make_request(
            "GET",
            "meta/bases",
            credentials=credentials,
            params=params if params else None,
        )

        bases = response.get("bases", [])
        result: Dict[str, Any] = {
            "bases": [
                {
                    "id": base.get("id"),
                    "name": base.get("name"),
                    "permissionLevel": base.get("permissionLevel"),
                }
                for base in bases
            ]
        }

        # Include pagination offset if present
        if "offset" in response:
            result["offset"] = response["offset"]

        logger.info("Found %d Airtable bases", len(bases))
        return result


class AirtableListTablesTool(AirtableAPIClient):
    """List tables in an Airtable base with field definitions."""

    name = "airtable_list_tables"
    description = "List all tables in an Airtable base, including field definitions and views."
    required_scopes = ["schema.bases:read"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_id": {
                    "type": "string",
                    "description": "Airtable base ID (e.g., 'appXXXXXXXXXXXXXX')",
                },
            },
            "required": ["base_id"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for base_id parameter."""
        return {
            "base_id": {
                "resource_type": "base",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "filter": {"provider": "airtable", "resource_type": "base"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "description": "List of tables in the base",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Table ID"},
                            "name": {"type": "string", "description": "Table name"},
                            "description": {"type": "string", "description": "Table description"},
                            "primaryFieldId": {"type": "string", "description": "ID of primary field"},
                            "fields": {
                                "type": "array",
                                "description": "List of field definitions",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                        "description": {"type": "string"},
                                    },
                                },
                            },
                            "views": {
                                "type": "array",
                                "description": "List of views",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "type": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "required": ["tables"],
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

        base_id = arguments.get("base_id")
        if not base_id:
            raise HTTPException(status_code=400, detail="Parameter 'base_id' is required")

        logger.info("Listing tables for Airtable base: %s", base_id)

        response = await self._make_request(
            "GET",
            f"meta/bases/{base_id}/tables",
            credentials=credentials,
        )

        tables = response.get("tables", [])
        result = {
            "tables": [
                {
                    "id": table.get("id"),
                    "name": table.get("name"),
                    "description": table.get("description"),
                    "primaryFieldId": table.get("primaryFieldId"),
                    "fields": [
                        {
                            "id": field.get("id"),
                            "name": field.get("name"),
                            "type": field.get("type"),
                            "description": field.get("description"),
                        }
                        for field in table.get("fields", [])
                    ],
                    "views": [
                        {
                            "id": view.get("id"),
                            "name": view.get("name"),
                            "type": view.get("type"),
                        }
                        for view in table.get("views", [])
                    ],
                }
                for table in tables
            ]
        }

        logger.info("Found %d tables in base %s", len(tables), base_id)
        return result
