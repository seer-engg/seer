"""
Google Docs helpers - schema definitions.
"""

from typing import Any, Dict


def _document_output_schema() -> Dict[str, Any]:
    """Schema for Google Docs Document resource."""
    return {
        "type": "object",
        "properties": {
            "documentId": {"type": "string", "description": "The ID of the document"},
            "title": {"type": "string", "description": "The title of the document"},
            "body": {
                "type": "object",
                "description": "The main body content of the document",
                "properties": {
                    "content": {
                        "type": "array",
                        "description": "The content of the body as structural elements",
                    },
                },
            },
            "documentStyle": {
                "type": "object",
                "description": "The style of the document",
            },
            "namedStyles": {
                "type": "object",
                "description": "The named styles of the document",
            },
            "revisionId": {
                "type": "string",
                "description": "The revision ID of the document",
            },
            "suggestionsViewMode": {
                "type": "string",
                "description": "The suggestions view mode applied to the document",
            },
        },
        "additionalProperties": True,
    }


def _batch_update_response_schema() -> Dict[str, Any]:
    """Schema for batchUpdate response."""
    return {
        "type": "object",
        "properties": {
            "documentId": {"type": "string", "description": "The ID of the document"},
            "replies": {
                "type": "array",
                "description": "The reply of the updates",
                "items": {"type": "object"},
            },
            "writeControl": {
                "type": "object",
                "description": "The updated write control after applying the request",
            },
        },
        "additionalProperties": True,
    }
