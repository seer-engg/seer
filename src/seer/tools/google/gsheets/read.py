"""
Google Sheets read operations - reading data from spreadsheets.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.google.base import GoogleAPIClient
from seer.tools.google.gsheets.helpers import (
    _value_range_output_schema,
    _batch_get_values_response_output_schema,
    _spreadsheet_output_schema,
)

logger = get_logger("shared.tools.gsheets.read")


class GoogleSheetsReadTool(GoogleAPIClient):
    """Read data from a single range in Google Sheets."""

    name = "google_sheets_read"
    description = "Read data from a Google Sheet range."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
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
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID"},
                "range": {"type": "string", "description": "A1 notation range", "default": "Sheet1"},
                "major_dimension": {"type": "string", "enum": ["ROWS", "COLUMNS"], "default": "ROWS"},
                "value_render_option": {"type": "string", "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"], "default": "FORMATTED_VALUE"},
                "date_time_render_option": {"type": "string", "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"], "default": "FORMATTED_STRING"},
            },
            "required": ["spreadsheet_id", "range"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _value_range_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1")

        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")

        params = {
            "majorDimension": arguments.get("major_dimension", "ROWS"),
            "valueRenderOption": arguments.get("value_render_option", "FORMATTED_VALUE"),
            "dateTimeRenderOption": arguments.get("date_time_render_option", "FORMATTED_STRING"),
        }

        logger.info("Reading from Google Sheet %s, range %s", spreadsheet_id, range_name)

        resp = await self._make_request(
            "GET",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}",
            access_token,
            params=params
        )
        return resp.json()


class GoogleSheetsBatchReadTool(GoogleAPIClient):
    """Read data from multiple ranges in Google Sheets."""

    name = "google_sheets_batch_read"
    description = "Read data from multiple ranges in a Google Sheet."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
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
                "ranges": {"type": "array", "items": {"type": "string"}, "description": "List of A1 notation ranges"},
                "major_dimension": {"type": "string", "enum": ["ROWS", "COLUMNS"], "default": "ROWS"},
                "value_render_option": {"type": "string", "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"], "default": "FORMATTED_VALUE"},
                "date_time_render_option": {"type": "string", "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"], "default": "FORMATTED_STRING"},
            },
            "required": ["spreadsheet_id", "ranges"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _batch_get_values_response_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        ranges = arguments.get("ranges", [])

        if not spreadsheet_id or not ranges:
            raise HTTPException(status_code=400, detail="spreadsheet_id and ranges are required")

        params = {
            "ranges": ranges,
            "majorDimension": arguments.get("major_dimension", "ROWS"),
            "valueRenderOption": arguments.get("value_render_option", "FORMATTED_VALUE"),
            "dateTimeRenderOption": arguments.get("date_time_render_option", "FORMATTED_STRING"),
        }

        resp = await self._make_request(
            "GET",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchGet",
            access_token,
            params=params
        )
        return resp.json()


class GoogleSheetsGetSpreadsheetTool(GoogleAPIClient):
    """Get Google Sheets spreadsheet metadata."""

    name = "google_sheets_get_spreadsheet"
    description = "Get metadata about a Google Sheets spreadsheet (sheets, properties)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
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
                "include_grid_data": {"type": "boolean", "default": False},
                "ranges": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["spreadsheet_id"]
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return _spreadsheet_output_schema()

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        spreadsheet_id = arguments.get("spreadsheet_id")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")

        params = {
            "includeGridData": arguments.get("include_grid_data", False)
        }

        if arguments.get("ranges"):
            params["ranges"] = arguments["ranges"]

        resp = await self._make_request(
            "GET",
            f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}",
            access_token,
            params=params
        )
        return resp.json()
