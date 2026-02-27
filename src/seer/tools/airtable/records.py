# pylint: disable=too-many-lines,too-complex
# Reason: Airtable record tools (list, create, update, delete) grouped together for cohesion; execute methods have many parameter validations
"""
Airtable record operations.

Provides tools for listing, creating, updating, and deleting records.
Airtable limits batch operations to 10 records per request.
"""
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.airtable.base import AirtableAPIClient
from seer.tools.credential_resolver import ResolvedCredentials

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.airtable.records")

# Airtable batch operation limit
MAX_RECORDS_PER_REQUEST = 10


class AirtableListRecordsTool(AirtableAPIClient):
    """List records from an Airtable table with filtering and sorting."""

    name = "airtable_list_records"
    description = "List records from an Airtable table. Supports filtering, sorting, and field selection."
    required_scopes = ["data.records:read"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_id": {
                    "type": "string",
                    "description": "Airtable base ID (e.g., 'appXXXXXXXXXXXXXX')",
                },
                "table_id_or_name": {
                    "type": "string",
                    "description": "Table ID or name (e.g., 'tblXXXXXXXXXXXXXX' or 'Tasks')",
                },
                "view": {
                    "type": "string",
                    "description": "View ID or name to use for filtering/sorting (optional)",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of field names to return (optional, returns all if not specified)",
                },
                "filter_by_formula": {
                    "type": "string",
                    "description": "Airtable formula to filter records (e.g., \"AND({Status}='Done', {Priority}='High')\")",
                },
                "max_records": {
                    "type": "integer",
                    "description": "Maximum number of records to return (default: 100, max: 100)",
                    "default": 100,
                    "maximum": 100,
                },
                "sort": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Field name to sort by"},
                            "direction": {"type": "string", "enum": ["asc", "desc"], "description": "Sort direction"},
                        },
                        "required": ["field"],
                    },
                    "description": "Sort configuration (optional)",
                },
                "offset": {
                    "type": "string",
                    "description": "Pagination offset from previous response (optional)",
                },
            },
            "required": ["base_id", "table_id_or_name"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for base_id and table parameters."""
        return {
            "base_id": {
                "resource_type": "base",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "filter": {"provider": "airtable", "resource_type": "base"},
            },
            "table_id_or_name": {
                "resource_type": "table",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "base_id",
                "filter": {"provider": "airtable", "resource_type": "table"},
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "List of records with flattened fields",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Record ID"},
                            "createdTime": {"type": "string", "description": "Record creation timestamp"},
                        },
                        "additionalProperties": True,
                    },
                },
                "offset": {
                    "type": "string",
                    "description": "Pagination offset for next page (if more records exist)",
                },
            },
            "required": ["records"],
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
        table_id_or_name = arguments.get("table_id_or_name")

        if not base_id:
            raise HTTPException(status_code=400, detail="Parameter 'base_id' is required")
        if not table_id_or_name:
            raise HTTPException(status_code=400, detail="Parameter 'table_id_or_name' is required")

        # Build query parameters
        params: Dict[str, Any] = {}

        if arguments.get("view"):
            params["view"] = arguments["view"]
        if arguments.get("fields"):
            # Airtable expects multiple 'fields[]' params
            params["fields[]"] = arguments["fields"]
        if arguments.get("filter_by_formula"):
            params["filterByFormula"] = arguments["filter_by_formula"]
        if arguments.get("max_records"):
            params["maxRecords"] = min(arguments["max_records"], 100)
        if arguments.get("sort"):
            # Airtable expects sort[0][field], sort[0][direction], etc.
            for i, sort_item in enumerate(arguments["sort"]):
                params[f"sort[{i}][field]"] = sort_item["field"]
                if "direction" in sort_item:
                    params[f"sort[{i}][direction]"] = sort_item["direction"]
        if arguments.get("offset"):
            params["offset"] = arguments["offset"]

        logger.info("Listing records from Airtable: base=%s, table=%s", base_id, table_id_or_name)

        response = await self._make_request(
            "GET",
            f"{base_id}/{table_id_or_name}",
            credentials=credentials,
            params=params if params else None,
        )

        records = response.get("records", [])
        result: Dict[str, Any] = {
            "records": self._format_records(records)
        }

        if "offset" in response:
            result["offset"] = response["offset"]

        logger.info("Retrieved %d records from %s/%s", len(records), base_id, table_id_or_name)
        return result


class AirtableCreateRecordTool(AirtableAPIClient):
    """Create one or more records in an Airtable table."""

    name = "airtable_create_record"
    description = "Create up to 10 records in an Airtable table. Use 'fields' for a single record or 'records' for batch creation."
    required_scopes = ["data.records:write"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_id": {
                    "type": "string",
                    "description": "Airtable base ID",
                },
                "table_id_or_name": {
                    "type": "string",
                    "description": "Table ID or name",
                },
                "fields": {
                    "type": "object",
                    "description": "Field values for a single record (use this OR 'records', not both)",
                    "additionalProperties": True,
                },
                "records": {
                    "type": "array",
                    "description": "Array of records for batch creation (max 10). Each item should have a 'fields' object.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fields": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["fields"],
                    },
                    "maxItems": 10,
                },
                "typecast": {
                    "type": "boolean",
                    "description": "If true, Airtable will try to convert string values to appropriate types",
                    "default": False,
                },
            },
            "required": ["base_id", "table_id_or_name"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for base_id and table parameters."""
        return {
            "base_id": {
                "resource_type": "base",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "filter": {"provider": "airtable", "resource_type": "base"},
            },
            "table_id_or_name": {
                "resource_type": "table",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "base_id",
                "filter": {"provider": "airtable", "resource_type": "table"},
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "Created records with their IDs",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "createdTime": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["records"],
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
        table_id_or_name = arguments.get("table_id_or_name")
        fields = arguments.get("fields")
        records = arguments.get("records")
        typecast = arguments.get("typecast", False)

        if not base_id:
            raise HTTPException(status_code=400, detail="Parameter 'base_id' is required")
        if not table_id_or_name:
            raise HTTPException(status_code=400, detail="Parameter 'table_id_or_name' is required")
        if not fields and not records:
            raise HTTPException(status_code=400, detail="Either 'fields' or 'records' is required")
        if fields and records:
            raise HTTPException(status_code=400, detail="Provide either 'fields' or 'records', not both")

        # Build records array
        if fields:
            records_to_create = [{"fields": fields}]
        else:
            records_to_create = records

        if len(records_to_create) > MAX_RECORDS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_RECORDS_PER_REQUEST} records can be created per request"
            )

        body: Dict[str, Any] = {"records": records_to_create}
        if typecast:
            body["typecast"] = True

        logger.info("Creating %d records in Airtable: base=%s, table=%s", len(records_to_create), base_id, table_id_or_name)

        response = await self._make_request(
            "POST",
            f"{base_id}/{table_id_or_name}",
            credentials=credentials,
            json_body=body,
        )

        created_records = response.get("records", [])
        logger.info("Created %d records in %s/%s", len(created_records), base_id, table_id_or_name)

        return {"records": self._format_records(created_records)}


class AirtableUpdateRecordTool(AirtableAPIClient):
    """Update one or more records in an Airtable table."""

    name = "airtable_update_record"
    description = "Update up to 10 records in an Airtable table. Supports partial updates (PATCH) and upsert mode."
    required_scopes = ["data.records:write"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_id": {
                    "type": "string",
                    "description": "Airtable base ID",
                },
                "table_id_or_name": {
                    "type": "string",
                    "description": "Table ID or name",
                },
                "record_id": {
                    "type": "string",
                    "description": "Record ID to update (use this OR 'records', not both)",
                },
                "fields": {
                    "type": "object",
                    "description": "Field values to update for a single record",
                    "additionalProperties": True,
                },
                "records": {
                    "type": "array",
                    "description": "Array of records for batch update (max 10). Each item should have 'id' and 'fields'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Record ID"},
                            "fields": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["id", "fields"],
                    },
                    "maxItems": 10,
                },
                "typecast": {
                    "type": "boolean",
                    "description": "If true, Airtable will try to convert string values to appropriate types",
                    "default": False,
                },
            },
            "required": ["base_id", "table_id_or_name"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for base_id and table parameters."""
        return {
            "base_id": {
                "resource_type": "base",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "filter": {"provider": "airtable", "resource_type": "base"},
            },
            "table_id_or_name": {
                "resource_type": "table",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "base_id",
                "filter": {"provider": "airtable", "resource_type": "table"},
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "Updated records",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "createdTime": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
            },
            "required": ["records"],
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
        table_id_or_name = arguments.get("table_id_or_name")
        record_id = arguments.get("record_id")
        fields = arguments.get("fields")
        records = arguments.get("records")
        typecast = arguments.get("typecast", False)

        if not base_id:
            raise HTTPException(status_code=400, detail="Parameter 'base_id' is required")
        if not table_id_or_name:
            raise HTTPException(status_code=400, detail="Parameter 'table_id_or_name' is required")

        # Build records array for batch update
        if record_id and fields:
            records_to_update = [{"id": record_id, "fields": fields}]
        elif records:
            records_to_update = records
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either ('record_id' and 'fields') or 'records'"
            )

        if len(records_to_update) > MAX_RECORDS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_RECORDS_PER_REQUEST} records can be updated per request"
            )

        body: Dict[str, Any] = {"records": records_to_update}
        if typecast:
            body["typecast"] = True

        logger.info("Updating %d records in Airtable: base=%s, table=%s", len(records_to_update), base_id, table_id_or_name)

        response = await self._make_request(
            "PATCH",
            f"{base_id}/{table_id_or_name}",
            credentials=credentials,
            json_body=body,
        )

        updated_records = response.get("records", [])
        logger.info("Updated %d records in %s/%s", len(updated_records), base_id, table_id_or_name)

        return {"records": self._format_records(updated_records)}


class AirtableDeleteRecordTool(AirtableAPIClient):
    """Delete one or more records from an Airtable table."""

    name = "airtable_delete_record"
    description = "Delete up to 10 records from an Airtable table."
    required_scopes = ["data.records:write"]
    integration_type = "airtable"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base_id": {
                    "type": "string",
                    "description": "Airtable base ID",
                },
                "table_id_or_name": {
                    "type": "string",
                    "description": "Table ID or name",
                },
                "record_id": {
                    "type": "string",
                    "description": "Single record ID to delete (use this OR 'record_ids', not both)",
                },
                "record_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of record IDs to delete (max 10)",
                    "maxItems": 10,
                },
            },
            "required": ["base_id", "table_id_or_name"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for base_id and table parameters."""
        return {
            "base_id": {
                "resource_type": "base",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "filter": {"provider": "airtable", "resource_type": "base"},
            },
            "table_id_or_name": {
                "resource_type": "table",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "base_id",
                "filter": {"provider": "airtable", "resource_type": "table"},
            },
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "description": "Deleted record information",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Deleted record ID"},
                            "deleted": {"type": "boolean", "description": "Deletion status"},
                        },
                    },
                },
            },
            "required": ["records"],
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
        table_id_or_name = arguments.get("table_id_or_name")
        record_id = arguments.get("record_id")
        record_ids = arguments.get("record_ids")

        if not base_id:
            raise HTTPException(status_code=400, detail="Parameter 'base_id' is required")
        if not table_id_or_name:
            raise HTTPException(status_code=400, detail="Parameter 'table_id_or_name' is required")
        if not record_id and not record_ids:
            raise HTTPException(status_code=400, detail="Either 'record_id' or 'record_ids' is required")
        if record_id and record_ids:
            raise HTTPException(status_code=400, detail="Provide either 'record_id' or 'record_ids', not both")

        # Build list of IDs to delete
        ids_to_delete = [record_id] if record_id else record_ids

        if len(ids_to_delete) > MAX_RECORDS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_RECORDS_PER_REQUEST} records can be deleted per request"
            )

        # Airtable expects multiple 'records[]' params for DELETE
        params = {"records[]": ids_to_delete}

        logger.info("Deleting %d records from Airtable: base=%s, table=%s", len(ids_to_delete), base_id, table_id_or_name)

        response = await self._make_request(
            "DELETE",
            f"{base_id}/{table_id_or_name}",
            credentials=credentials,
            params=params,
        )

        deleted_records = response.get("records", [])
        logger.info("Deleted %d records from %s/%s", len(deleted_records), base_id, table_id_or_name)

        return {
            "records": [
                {"id": record.get("id"), "deleted": record.get("deleted", True)}
                for record in deleted_records
            ]
        }
