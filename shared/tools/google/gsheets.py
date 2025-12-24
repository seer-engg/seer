"""
Google Sheets Tool

Tool for writing data to Google Sheets using Google Sheets API v4.
"""
from typing import Any, Dict, Optional
import json
import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.google_sheets")


class GoogleSheetsWriteTool(BaseTool):
    """Tool for writing data to Google Sheets."""
    
    name = "google_sheets_write"
    description = "Write data to a Google Sheet. Requires spreadsheet ID and range."
    required_scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        # TODO: it is hack for now to allow writing to Google Sheets, we need to find a better way to do this.
        "https://www.googleapis.com/auth/drive.file"
        ]
    integration_type = "google_sheets"
    provider = "google"
    
    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource browsing for spreadsheet_id."""
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
        """Get JSON schema for Google Sheets write tool parameters."""
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {
                    "type": "string",
                    "description": "Google Sheets spreadsheet ID (from URL)"
                    
                },
                "range": {
                    "type": "string",
                    "description": "A1 notation range (e.g., 'Sheet1!A1' or 'Sheet1!A1:B10')",
                    "default": "Sheet1!A1"
                },
                "values": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "description": "2D array of values to write (rows x columns)"
                },
                "value_input_option": {
                    "type": "string",
                    "description": "How input data should be interpreted",
                    "enum": ["RAW", "USER_ENTERED"],
                    "default": "USER_ENTERED"
                }
            },
            "required": ["spreadsheet_id", "range", "values"]
        }
    
    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        """
        Execute Google Sheets write tool.
        
        Args:
            access_token: OAuth access token (required)
            arguments: Tool arguments
        
        Returns:
            Google Sheets API response
        """
        if not access_token:
            raise HTTPException(
                status_code=401,
                detail="Google Sheets tool requires OAuth access token"
            )
        
        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1!A1")
        values = arguments.get("values")
        value_input_option = arguments.get("value_input_option", "USER_ENTERED")
        
        if not spreadsheet_id:
            raise HTTPException(
                status_code=400,
                detail="spreadsheet_id is required"
            )
        
        if not values:
            raise HTTPException(
                status_code=400,
                detail="values is required"
            )
        
        # Parse values if it's a string (e.g., from workflow config)
        if isinstance(values, str):
            try:
                values = json.loads(values)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON format for 'values' parameter: {str(e)}. Expected a 2D array like [[\"a\", \"b\"], [\"c\", \"d\"]]"
                )
        
        # Validate that values is a 2D array
        if not isinstance(values, list):
            raise HTTPException(
                status_code=400,
                detail="'values' must be a 2D array (list of lists)"
            )
        
        # Use Google Sheets API v4
        # Note: valueInputOption MUST be a query parameter, not in the request body
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
        params = {
            "valueInputOption": value_input_option
        }
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        body = {
            "values": values
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Writing to Google Sheet {spreadsheet_id}, range {range_name}")
                
                response = await http_client.put(url, headers=headers, params=params, json=body)
                
                if response.status_code == 401:
                    raise HTTPException(
                        status_code=401,
                        detail="Google Sheets API authentication failed. Token may be expired or invalid."
                    )
                
                if response.status_code == 403:
                    raise HTTPException(
                        status_code=403,
                        detail="Permission denied. Ensure the spreadsheet is accessible and the OAuth token has write permissions."
                    )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Successfully wrote {result.get('updatedCells', 0)} cells to Google Sheet")
                return result
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Google Sheets API error: {e.response.text[:500]}"
            )
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(
                status_code=504,
                detail="Google Sheets API request timed out"
            )
        except Exception as e:
            logger.exception(f"Unexpected error writing to Google Sheets: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error writing to Google Sheets: {str(e)}"
            )


# Register the tool
_google_sheets_tool = GoogleSheetsWriteTool()
register_tool(_google_sheets_tool)



"""
Google Sheets Tools

A collection of tools for Google Sheets integration using Google Sheets API v4.

Includes:
- Read single range (values.get)
- Read multiple ranges (values.batchGet)
- Append values (values.append)
- Clear values (values.clear)
- Batch write values (values.batchUpdate)
- Get spreadsheet metadata (spreadsheets.get)
- Create spreadsheet (spreadsheets.create)
- Batch update spreadsheet (spreadsheets.batchUpdate) for formatting/structure ops
"""
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.google_sheets")


# ----------------------------
# Helpers
# ----------------------------
def _require_access_token(access_token: Optional[str]) -> None:
    if not access_token:
        raise HTTPException(status_code=401, detail="Google Sheets tool requires OAuth access token")


def _handle_common_errors(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Google Sheets API authentication failed. Token may be expired or invalid.",
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail="Permission denied. Ensure the spreadsheet is accessible and the OAuth token has required scopes.",
        )


def _values_schema(description: str) -> Dict[str, Any]:
    # Sheets ValueRange supports strings, numbers, booleans, and nulls.
    return {
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                ]
            },
        },
        "description": description,
    }


# ----------------------------
# Read single range
# ----------------------------
class GoogleSheetsReadTool(BaseTool):
    """Tool for reading data from a Google Sheet range."""

    name = "google_sheets_read"
    description = "Read values from a Google Sheet range. Requires spreadsheet ID and range."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource browsing for spreadsheet_id."""
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
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "range": {
                    "type": "string",
                    "description": "A1 notation range (e.g., 'Sheet1!A1' or 'Sheet1!A1:B10')",
                    "default": "Sheet1!A1",
                },
                "major_dimension": {
                    "type": "string",
                    "description": "Dimension that results should use",
                    "enum": ["ROWS", "COLUMNS"],
                    "default": "ROWS",
                },
                "value_render_option": {
                    "type": "string",
                    "description": "How values should be rendered in the response",
                    "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"],
                    "default": "FORMATTED_VALUE",
                },
                "date_time_render_option": {
                    "type": "string",
                    "description": "How dates/times should be rendered (ignored if value_render_option=FORMATTED_VALUE)",
                    "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"],
                    "default": "SERIAL_NUMBER",
                },
            },
            "required": ["spreadsheet_id", "range"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1!A1")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")

        params = {
            "majorDimension": arguments.get("major_dimension", "ROWS"),
            "valueRenderOption": arguments.get("value_render_option", "FORMATTED_VALUE"),
            "dateTimeRenderOption": arguments.get("date_time_render_option", "SERIAL_NUMBER"),
        }

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Reading Google Sheet {spreadsheet_id}, range {range_name}")
                response = await http_client.get(url, headers=headers, params=params)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error reading from Google Sheets: {e}")
            raise HTTPException(status_code=500, detail=f"Error reading from Google Sheets: {str(e)}")


# ----------------------------
# Read multiple ranges
# ----------------------------
class GoogleSheetsBatchReadTool(BaseTool):
    """Tool for reading values from multiple ranges in a spreadsheet."""

    name = "google_sheets_batch_read"
    description = "Read values from multiple ranges in a Google Sheet (batchGet)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "ranges": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of A1 ranges (e.g., ['Sheet1!A1:B2','Sheet2!C1:D5'])",
                },
                "major_dimension": {
                    "type": "string",
                    "description": "Dimension that results should use",
                    "enum": ["ROWS", "COLUMNS"],
                    "default": "ROWS",
                },
                "value_render_option": {
                    "type": "string",
                    "description": "How values should be rendered in the response",
                    "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"],
                    "default": "FORMATTED_VALUE",
                },
                "date_time_render_option": {
                    "type": "string",
                    "description": "How dates/times should be rendered (ignored if value_render_option=FORMATTED_VALUE)",
                    "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"],
                    "default": "SERIAL_NUMBER",
                },
            },
            "required": ["spreadsheet_id", "ranges"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        ranges = arguments.get("ranges")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")
        if not ranges or not isinstance(ranges, list):
            raise HTTPException(status_code=400, detail="ranges must be a non-empty list")

        params = {
            "majorDimension": arguments.get("major_dimension", "ROWS"),
            "valueRenderOption": arguments.get("value_render_option", "FORMATTED_VALUE"),
            "dateTimeRenderOption": arguments.get("date_time_render_option", "SERIAL_NUMBER"),
            "ranges": ranges,  # repeated query param
        }

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchGet"
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Batch reading Google Sheet {spreadsheet_id}, ranges={len(ranges)}")
                response = await http_client.get(url, headers=headers, params=params)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error batch reading from Google Sheets: {e}")
            raise HTTPException(status_code=500, detail=f"Error batch reading from Google Sheets: {str(e)}")


# ----------------------------
# Append values
# ----------------------------
class GoogleSheetsAppendTool(BaseTool):
    """Tool for appending data to a Google Sheet (adds rows)."""

    name = "google_sheets_append"
    description = "Append values to a Google Sheet (values.append)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "range": {
                    "type": "string",
                    "description": "A1 notation range used to detect a table (e.g., 'Sheet1!A1:D1' or 'Sheet1!A:D')",
                    "default": "Sheet1!A1",
                },
                "values": _values_schema("2D array of values to append (rows x columns)"),
                "value_input_option": {
                    "type": "string",
                    "description": "How input data should be interpreted",
                    "enum": ["RAW", "USER_ENTERED"],
                    "default": "USER_ENTERED",
                },
                "insert_data_option": {
                    "type": "string",
                    "description": "How the input data should be inserted",
                    "enum": ["OVERWRITE", "INSERT_ROWS"],
                    "default": "INSERT_ROWS",
                },
                "include_values_in_response": {
                    "type": "boolean",
                    "description": "Whether to include the appended values in the response",
                    "default": False,
                },
                "response_value_render_option": {
                    "type": "string",
                    "description": "How values in the response should be rendered",
                    "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"],
                    "default": "FORMATTED_VALUE",
                },
                "response_date_time_render_option": {
                    "type": "string",
                    "description": "How dates/times in the response should be rendered",
                    "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"],
                    "default": "SERIAL_NUMBER",
                },
                "major_dimension": {
                    "type": "string",
                    "description": "Major dimension for the input ValueRange",
                    "enum": ["ROWS", "COLUMNS"],
                    "default": "ROWS",
                },
            },
            "required": ["spreadsheet_id", "range", "values"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1!A1")
        values = arguments.get("values")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")
        if values is None:
            raise HTTPException(status_code=400, detail="values is required")

        params = {
            "valueInputOption": arguments.get("value_input_option", "USER_ENTERED"),
            "insertDataOption": arguments.get("insert_data_option", "INSERT_ROWS"),
            "includeValuesInResponse": arguments.get("include_values_in_response", False),
            "responseValueRenderOption": arguments.get("response_value_render_option", "FORMATTED_VALUE"),
            "responseDateTimeRenderOption": arguments.get("response_date_time_render_option", "SERIAL_NUMBER"),
        }

        body = {
            "majorDimension": arguments.get("major_dimension", "ROWS"),
            "values": values,
        }

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:append"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Appending to Google Sheet {spreadsheet_id}, range {range_name}")
                response = await http_client.post(url, headers=headers, params=params, json=body)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error appending to Google Sheets: {e}")
            raise HTTPException(status_code=500, detail=f"Error appending to Google Sheets: {str(e)}")


# ----------------------------
# Clear values in range
# ----------------------------
class GoogleSheetsClearTool(BaseTool):
    """Tool for clearing values from a Google Sheet range."""

    name = "google_sheets_clear"
    description = "Clear values from a Google Sheet range (values.clear)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "range": {
                    "type": "string",
                    "description": "A1 notation range to clear (e.g., 'Sheet1!A1:B10')",
                    "default": "Sheet1!A1",
                },
            },
            "required": ["spreadsheet_id", "range"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        range_name = arguments.get("range", "Sheet1!A1")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}:clear"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Clearing Google Sheet {spreadsheet_id}, range {range_name}")
                response = await http_client.post(url, headers=headers, json={})
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error clearing Google Sheets: {e}")
            raise HTTPException(status_code=500, detail=f"Error clearing Google Sheets: {str(e)}")


# ----------------------------
# Batch write values (multiple ranges)
# ----------------------------
class GoogleSheetsBatchWriteTool(BaseTool):
    """Tool for writing values to multiple ranges in one call (values.batchUpdate)."""

    name = "google_sheets_batch_write"
    description = "Write values to multiple ranges in a Google Sheet (values.batchUpdate)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "value_input_option": {
                    "type": "string",
                    "description": "How input data should be interpreted",
                    "enum": ["RAW", "USER_ENTERED"],
                    "default": "USER_ENTERED",
                },
                "data": {
                    "type": "array",
                    "description": "List of ValueRange objects (each with a range and values).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "range": {"type": "string", "description": "A1 range for this update"},
                            "major_dimension": {
                                "type": "string",
                                "enum": ["ROWS", "COLUMNS"],
                                "default": "ROWS",
                            },
                            "values": _values_schema("2D array of values for this range"),
                        },
                        "required": ["range", "values"],
                    },
                },
                "include_values_in_response": {
                    "type": "boolean",
                    "description": "Whether to include the updated values in the response",
                    "default": False,
                },
                "response_value_render_option": {
                    "type": "string",
                    "description": "How values in the response should be rendered",
                    "enum": ["FORMATTED_VALUE", "UNFORMATTED_VALUE", "FORMULA"],
                    "default": "FORMATTED_VALUE",
                },
                "response_date_time_render_option": {
                    "type": "string",
                    "description": "How dates/times in the response should be rendered",
                    "enum": ["SERIAL_NUMBER", "FORMATTED_STRING"],
                    "default": "SERIAL_NUMBER",
                },
            },
            "required": ["spreadsheet_id", "data"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        data = arguments.get("data")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")
        if not data or not isinstance(data, list):
            raise HTTPException(status_code=400, detail="data must be a non-empty list")

        request_data = []
        for item in data:
            if not isinstance(item, dict) or not item.get("range") or item.get("values") is None:
                raise HTTPException(status_code=400, detail="Each data item must include 'range' and 'values'")
            request_data.append(
                {
                    "range": item["range"],
                    "majorDimension": item.get("major_dimension", "ROWS"),
                    "values": item["values"],
                }
            )

        body = {
            "valueInputOption": arguments.get("value_input_option", "USER_ENTERED"),
            "data": request_data,
            "includeValuesInResponse": arguments.get("include_values_in_response", False),
            "responseValueRenderOption": arguments.get("response_value_render_option", "FORMATTED_VALUE"),
            "responseDateTimeRenderOption": arguments.get("response_date_time_render_option", "SERIAL_NUMBER"),
        }

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values:batchUpdate"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Batch writing Google Sheet {spreadsheet_id}, updates={len(request_data)}")
                response = await http_client.post(url, headers=headers, json=body)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error batch writing to Google Sheets: {e}")
            raise HTTPException(status_code=500, detail=f"Error batch writing to Google Sheets: {str(e)}")


# ----------------------------
# Get spreadsheet metadata (optionally include grid data)
# ----------------------------
class GoogleSheetsGetSpreadsheetTool(BaseTool):
    """Tool for retrieving spreadsheet metadata (and optionally grid data)."""

    name = "google_sheets_get_spreadsheet"
    description = "Get spreadsheet metadata (and optional grid data) via spreadsheets.get."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "include_grid_data": {
                    "type": "boolean",
                    "description": "If true, includes grid data (can be large). Prefer fields/ranges for partial reads.",
                    "default": False,
                },
                "ranges": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional A1 ranges to limit returned data (e.g., ['Sheet1!A1:D10']).",
                },
                "fields": {
                    "type": "string",
                    "description": "Optional partial response field mask (e.g., 'properties.title,sheets.properties').",
                },
            },
            "required": ["spreadsheet_id"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")

        params: Dict[str, Any] = {
            "includeGridData": arguments.get("include_grid_data", False),
        }
        ranges = arguments.get("ranges")
        if ranges:
            params["ranges"] = ranges  # repeated query param
        fields = arguments.get("fields")
        if fields:
            params["fields"] = fields

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}"
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Fetching spreadsheet metadata {spreadsheet_id}")
                response = await http_client.get(url, headers=headers, params=params)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error fetching spreadsheet metadata: {e}")
            raise HTTPException(status_code=500, detail=f"Error fetching spreadsheet metadata: {str(e)}")


# ----------------------------
# Create spreadsheet
# ----------------------------
class GoogleSheetsCreateSpreadsheetTool(BaseTool):
    """Tool for creating a new spreadsheet."""

    name = "google_sheets_create_spreadsheet"
    description = "Create a new Google Sheet (spreadsheets.create)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"
    provider = "google"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Spreadsheet title", "default": "New Spreadsheet"},
                "spreadsheet": {
                    "type": "object",
                    "description": (
                        "Optional raw Spreadsheet resource body. If provided, it will be merged with title. "
                        "Use this for advanced creation (multiple sheets, properties, etc.)."
                    ),
                },
            },
            "required": [],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        title = arguments.get("title", "New Spreadsheet")
        spreadsheet_obj = arguments.get("spreadsheet") or {}

        # Ensure title is set
        properties = spreadsheet_obj.get("properties") if isinstance(spreadsheet_obj, dict) else None
        if not isinstance(properties, dict):
            properties = {}
        properties.setdefault("title", title)

        body = dict(spreadsheet_obj) if isinstance(spreadsheet_obj, dict) else {}
        body["properties"] = properties

        url = "https://sheets.googleapis.com/v4/spreadsheets"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Creating spreadsheet with title='{properties.get('title')}'")
                response = await http_client.post(url, headers=headers, json=body)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error creating spreadsheet: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating spreadsheet: {str(e)}")


# ----------------------------
# BatchUpdate spreadsheet (formatting/structure)
# ----------------------------
class GoogleSheetsBatchUpdateSpreadsheetTool(BaseTool):
    """Tool for performing structural/formatting updates using spreadsheets.batchUpdate."""

    name = "google_sheets_batch_update_spreadsheet"
    description = "Batch update a spreadsheet for formatting/structure (spreadsheets.batchUpdate)."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "google_sheets"
    provider = "google"

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "spreadsheet_id": {
                "resource_type": "google_spreadsheet",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
            }
        }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "spreadsheet_id": {"type": "string", "description": "Google Sheets spreadsheet ID (from URL)"},
                "requests": {
                    "type": "array",
                    "description": (
                        "List of BatchUpdate requests (raw). "
                        "Examples: addSheet, deleteSheet, repeatCell, updateSheetProperties, etc."
                    ),
                    "items": {"type": "object"},
                },
                "include_spreadsheet_in_response": {
                    "type": "boolean",
                    "description": "Whether to include the updated Spreadsheet in the response",
                    "default": False,
                },
                "response_include_grid_data": {
                    "type": "boolean",
                    "description": "If including spreadsheet in response, whether to include grid data",
                    "default": False,
                },
                "response_ranges": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "If including spreadsheet in response, limit returned grid data to these ranges",
                },
            },
            "required": ["spreadsheet_id", "requests"],
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any]) -> Any:
        _require_access_token(access_token)

        spreadsheet_id = arguments.get("spreadsheet_id")
        requests = arguments.get("requests")
        if not spreadsheet_id:
            raise HTTPException(status_code=400, detail="spreadsheet_id is required")
        if not requests or not isinstance(requests, list):
            raise HTTPException(status_code=400, detail="requests must be a non-empty list")

        body: Dict[str, Any] = {
            "requests": requests,
            "includeSpreadsheetInResponse": arguments.get("include_spreadsheet_in_response", False),
            "responseIncludeGridData": arguments.get("response_include_grid_data", False),
        }
        response_ranges = arguments.get("response_ranges")
        if response_ranges:
            body["responseRanges"] = response_ranges

        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}:batchUpdate"
        headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Batch updating spreadsheet {spreadsheet_id}, requests={len(requests)}")
                response = await http_client.post(url, headers=headers, json=body)
                _handle_common_errors(response)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Google Sheets API error: {e.response.status_code} - {e.response.text[:500]}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Google Sheets API error: {e.response.text[:500]}")
        except httpx.TimeoutException:
            logger.error("Google Sheets API request timed out")
            raise HTTPException(status_code=504, detail="Google Sheets API request timed out")
        except Exception as e:
            logger.exception(f"Unexpected error batch updating spreadsheet: {e}")
            raise HTTPException(status_code=500, detail=f"Error batch updating spreadsheet: {str(e)}")


# ----------------------------
# Register tools
# ----------------------------
register_tool(GoogleSheetsReadTool())
register_tool(GoogleSheetsBatchReadTool())
register_tool(GoogleSheetsAppendTool())
register_tool(GoogleSheetsClearTool())
register_tool(GoogleSheetsBatchWriteTool())
register_tool(GoogleSheetsGetSpreadsheetTool())
register_tool(GoogleSheetsCreateSpreadsheetTool())
register_tool(GoogleSheetsBatchUpdateSpreadsheetTool())
