"""
Google Sheets helpers - schema definitions.
"""

from typing import Any, Dict


def _values_schema(description: str) -> Dict[str, Any]:
    """Schema for 2D array of cell values."""
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


def _cell_schema() -> Dict[str, Any]:
    """JSON schema for a single cell value."""
    return {
        "oneOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }


def _value_range_output_schema() -> Dict[str, Any]:
    """Schema for ValueRange response."""
    return {
        "type": "object",
        "properties": {
            "range": {"type": "string"},
            "majorDimension": {"type": "string"},
            "values": _values_schema("2D array of cell values"),
        },
        "additionalProperties": True,
    }


def _update_values_response_output_schema() -> Dict[str, Any]:
    """Schema for update values response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "updatedRange": {"type": "string"},
            "updatedRows": {"type": "integer"},
            "updatedColumns": {"type": "integer"},
            "updatedCells": {"type": "integer"},
        },
        "additionalProperties": True,
    }


def _batch_get_values_response_output_schema() -> Dict[str, Any]:
    """Schema for batch get values response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "valueRanges": {
                "type": "array",
                "items": _value_range_output_schema(),
            },
        },
        "additionalProperties": True,
    }


def _append_values_response_output_schema() -> Dict[str, Any]:
    """Schema for append values response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "tableRange": {"type": "string"},
            "updates": _update_values_response_output_schema(),
        },
        "additionalProperties": True,
    }


def _clear_values_response_output_schema() -> Dict[str, Any]:
    """Schema for clear values response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "clearedRange": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _batch_update_values_response_output_schema() -> Dict[str, Any]:
    """Schema for batch update values response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "totalUpdatedRows": {"type": "integer"},
            "totalUpdatedColumns": {"type": "integer"},
            "totalUpdatedCells": {"type": "integer"},
            "totalUpdatedSheets": {"type": "integer"},
            "responses": {
                "type": "array",
                "items": _update_values_response_output_schema(),
            },
        },
        "additionalProperties": True,
    }


def _spreadsheet_output_schema() -> Dict[str, Any]:
    """Schema for Spreadsheet resource."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "properties": {"type": "object"},
            "sheets": {"type": "array"},
            "spreadsheetUrl": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _batch_update_spreadsheet_response_output_schema() -> Dict[str, Any]:
    """Schema for batch update spreadsheet response."""
    return {
        "type": "object",
        "properties": {
            "spreadsheetId": {"type": "string"},
            "replies": {"type": "array"},
            "updatedSpreadsheet": _spreadsheet_output_schema(),
        },
        "additionalProperties": True,
    }
