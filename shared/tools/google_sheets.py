"""
Google Sheets Tool

Tool for writing data to Google Sheets using Google Sheets API v4.
"""
from typing import Any, Dict, Optional
import httpx
from fastapi import HTTPException

from shared.tools.base import BaseTool, register_tool
from shared.logger import get_logger

logger = get_logger("shared.tools.google_sheets")


class GoogleSheetsWriteTool(BaseTool):
    """Tool for writing data to Google Sheets."""
    
    name = "google_sheets_write"
    description = "Write data to a Google Sheet. Requires spreadsheet ID and range."
    required_scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    integration_type = "googlesheets"
    
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
        
        # Use Google Sheets API v4
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{range_name}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        body = {
            "values": values,
            "valueInputOption": value_input_option
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                logger.info(f"Writing to Google Sheet {spreadsheet_id}, range {range_name}")
                
                response = await http_client.put(url, headers=headers, json=body)
                
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

