"""
JSON Schema definitions for file inputs and outputs in workflows.

This module provides standardized schemas for tools that accept or return files.
Tools should use these schemas in their get_parameters_schema() and get_output_schema()
methods to ensure consistent file handling across the workflow system.

Two input modes are supported:
1. Dynamic: File reference from parent node output (WorkflowFileRef)
2. Static: File selected from user's storage by file_id (static_file_ref)
"""

from __future__ import annotations

from typing import Any

from seer.core.files.models import WORKFLOW_FILE_REF_TYPE

# Type marker for static file references (user-uploaded files selected at workflow design time)
STATIC_FILE_REF_TYPE = "static_file_ref"


def is_static_file_ref(value: Any) -> bool:
    """
    Check if a value is a static file reference.

    Static file references are used to select files from user storage
    at workflow design time, as opposed to dynamic file references
    that come from parent node outputs.

    Args:
        value: Any value to check.

    Returns:
        True if the value is a static file reference dict.
    """
    return isinstance(value, dict) and value.get("_type") == STATIC_FILE_REF_TYPE


# Schema for file inputs that accept both dynamic and static references
FILE_INPUT_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "title": "Dynamic file reference",
            "description": "File reference from parent node output (e.g., ${download.file})",
            "properties": {
                "_type": {"const": WORKFLOW_FILE_REF_TYPE},
                "file_id": {"type": "string", "description": "Unique file identifier"},
                "storage_path": {"type": "string", "description": "S3 storage location"},
                "filename": {"type": "string", "description": "Original filename"},
                "mime_type": {"type": "string", "description": "MIME type"},
                "size_bytes": {"type": "integer", "description": "File size in bytes"},
                "workflow_run_id": {"type": "string"},
                "created_at": {"type": "string", "format": "date-time"},
                "md5_hash": {"type": ["string", "null"]},
            },
            "required": ["_type", "file_id", "storage_path", "filename", "mime_type", "size_bytes"],
        },
        {
            "type": "object",
            "title": "Static file reference",
            "description": "File from user's storage, selected at workflow design time",
            "properties": {
                "_type": {"const": STATIC_FILE_REF_TYPE},
                "file_id": {
                    "type": "string",
                    "description": "File ID from user's storage (validated at compile time)",
                },
            },
            "required": ["_type", "file_id"],
        },
    ],
}


# Schema for file outputs (always returns WorkflowFileRef)
FILE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "File reference",
    "description": "Reference to a file stored in workflow file system",
    "properties": {
        "_type": {"const": WORKFLOW_FILE_REF_TYPE},
        "file_id": {"type": "string", "description": "Unique file identifier"},
        "storage_path": {"type": "string", "description": "S3 storage location"},
        "filename": {"type": "string", "description": "Original filename"},
        "mime_type": {"type": "string", "description": "MIME type"},
        "size_bytes": {"type": "integer", "description": "File size in bytes"},
        "workflow_run_id": {"type": "string"},
        "created_at": {"type": "string", "format": "date-time"},
        "md5_hash": {"type": ["string", "null"]},
    },
    "required": ["_type", "file_id", "filename", "mime_type", "size_bytes"],
}


def get_file_input_property(description: str = "File input") -> dict[str, Any]:
    """
    Get a file input property schema with custom description.

    This is a convenience function for tools that need a single file input.

    Args:
        description: Description for the file input property.

    Returns:
        Property schema for use in tool parameters.
    """
    schema = FILE_INPUT_SCHEMA.copy()
    schema["description"] = description
    return schema


def get_file_array_input_property(
    description: str = "File inputs",
    min_items: int = 0,
    max_items: int | None = None,
) -> dict[str, Any]:
    """
    Get a file array input property schema.

    This is useful for tools that accept multiple files (e.g., email attachments).

    Args:
        description: Description for the file array property.
        min_items: Minimum number of files required.
        max_items: Maximum number of files allowed (None for unlimited).

    Returns:
        Property schema for array of file inputs.
    """
    schema: dict[str, Any] = {
        "type": "array",
        "description": description,
        "items": FILE_INPUT_SCHEMA,
        "minItems": min_items,
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


__all__ = [
    "STATIC_FILE_REF_TYPE",
    "FILE_INPUT_SCHEMA",
    "FILE_OUTPUT_SCHEMA",
    "is_static_file_ref",
    "get_file_input_property",
    "get_file_array_input_property",
]
