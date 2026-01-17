"""
Google Sheets write operations - writing and updating data.
"""

import json
from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gsheets.helpers import (
    _update_values_response_output_schema,
    _append_values_response_output_schema,
    _clear_values_response_output_schema,
    _batch_update_values_response_output_schema,
    _spreadsheet_output_schema,
    _batch_update_spreadsheet_response_output_schema,
)

logger = get_logger("shared.tools.gsheets.write")


class GoogleSheetsWriteTool(GoogleAPIClient):
    """Write data to Google Sheets."""

    name = "google_sheets_write"
    description = "Write data to a Google Sheet. Requires spreadsheet ID and range."
    required_scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.metadata.readonly"
    ]
    integration_type = "google_sheets"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
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
                "spreadsheet_id": {"type": "string"},
                "range": {"type": "string", "default": "Sheet1!A1"},
                "values": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "2D array of values"
                },
                "value_input_option": {"type": "string", "enum": ["RAW", "USER_ENTERED"], "default": "USER_ENTERED"}
            },
            "required": ["spreadsheet_id", "range", "values"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _update_values_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1!A1")
        values = arguments.get("values")

        if not spreadsheet_id or not values:
            raise HTTPException(status_code=400, detail="spreadsheet_id and values are required")

        # Parse values if string
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        params = {"valueInputOption": arguments.get("value_input_option", "USER_ENTERED")}
        body = {"values": values}

        logger.info("Writing to Google Sheet %s, range %s", spreadsheet_id, range_name)

        resp = await self._make_request(
            "PUT",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}",
            access_token,
            params=params,
            json_body=body
        )
        return resp.json()


class GoogleSheetsAppendTool(GoogleAPIClient):
    """Append data to Google Sheets."""

    name = "google_sheets_append"
    description = "Append data to a Google Sheet (adds rows after existing data)."
    required_scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.metadata.readonly"
    ]
    integration_type = "google_sheets"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
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
                "spreadsheet_id": {"type": "string"},
                "range": {"type": "string", "default": "Sheet1"},
                "values": {"type": "array", "items": {"type": "array"}},
                "value_input_option": {"type": "string", "enum": ["RAW", "USER_ENTERED"], "default": "USER_ENTERED"},
                "insert_data_option": {"type": "string", "enum": ["OVERWRITE", "INSERT_ROWS"], "default": "INSERT_ROWS"},
            },
            "required": ["spreadsheet_id", "range", "values"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _append_values_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1")
        values = arguments.get("values")

        if not spreadsheet_id or not values:
            raise HTTPException(status_code=400, detail="spreadsheet_id and values are required")

        params = {
            "valueInputOption": arguments.get("value_input_option", "USER_ENTERED"),
            "insertDataOption": arguments.get("insert_data_option", "INSERT_ROWS"),
        }
        body = {"values": values}

        resp = await self._make_request(
            "POST",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:append",
            access_token,
            params=params,
            json_body=body
        )
        return resp.json()


class GoogleSheetsClearTool(GoogleAPIClient):
    """Clear data from Google Sheets range."""

    name = "google_sheets_clear"
    description = "Clear data from a Google Sheet range."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string"},
                "range": {"type": "string"},
            },
            "required": ["spreadsheet_id", "range"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _clear_values_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range")

        if not spreadsheet_id or not range_name:
            raise HTTPException(status_code=400, detail="spreadsheet_id and range are required")

        resp = await self._make_request(
            "POST",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:clear",
            access_token
        )
        return resp.json()


class GoogleSheetsBatchWriteTool(GoogleAPIClient):
    """Batch write to multiple ranges in Google Sheets."""

    name = "google_sheets_batch_write"
    description = "Write data to multiple ranges in a Google Sheet."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string"},
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "range": {"type": "string"},
                            "values": {"type": "array"}
                        }
                    }
                },
                "value_input_option": {"type": "string", "enum": ["RAW", "USER_ENTERED"], "default": "USER_ENTERED"},
            },
            "required": ["spreadsheet_id", "data"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _batch_update_values_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        data = arguments.get("data")

        if not spreadsheet_id or not data:
            raise HTTPException(status_code=400, detail="spreadsheet_id and data are required")

        body = {
            "valueInputOption": arguments.get("value_input_option", "USER_ENTERED"),
            "data": data
        }

        resp = await self._make_request(
            "POST",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchUpdate",
            access_token,
            json_body=body
        )
        return resp.json()


class GoogleSheetsCreateSpreadsheetTool(GoogleAPIClient):
    """Create a new Google Sheets spreadsheet."""

    name = "google_sheets_create_spreadsheet"
    description = "Create a new Google Sheets spreadsheet."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Spreadsheet title"},
                "sheets": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Sheet properties"
                },
            },
            "required": ["title"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _spreadsheet_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        title = arguments.get("title")
        if not title:
            raise HTTPException(status_code=400, detail="title is required")

        body = {"properties": {"title": title}}
        if arguments.get("sheets"):
            body["sheets"] = arguments["sheets"]

        resp = await self._make_request(
            "POST",
            "https://sheets.googleapis.com/v4/spreadsheets",
            access_token,
            json_body=body
        )
        return resp.json()


class GoogleSheetsBatchUpdateSpreadsheetTool(GoogleAPIClient):
    """Batch update Google Sheets spreadsheet (formatting, structure)."""

    name = "google_sheets_batch_update_spreadsheet"
    description = "Batch update spreadsheet formatting and structure."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string"},
                "requests": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Array of Request objects"
                },
                "include_spreadsheet_in_response": {"type": "boolean", "default": False},
            },
            "required": ["spreadsheet_id", "requests"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _batch_update_spreadsheet_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        requests = arguments.get("requests")

        if not spreadsheet_id or not requests:
            raise HTTPException(status_code=400, detail="spreadsheet_id and requests are required")

        body = {
            "requests": requests,
            "includeSpreadsheetInResponse": arguments.get("include_spreadsheet_in_response", False),
        }

        resp = await self._make_request(
            "POST",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate",
            access_token,
            json_body=body
        )
        return resp.json()
