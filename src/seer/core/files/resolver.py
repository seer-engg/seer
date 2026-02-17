"""
Unified file resolution for workflow tools.

This module provides the core file resolution logic that tools use to convert
file inputs (either dynamic WorkflowFileRef or static_file_ref) to actual file bytes.

The resolver handles:
1. WorkflowFileRef (from parent node outputs) - resolved via WorkflowFileSystem
2. static_file_ref (from user storage) - resolved by looking up the file in database
3. Legacy base64 strings - REJECTED with clear error message

This ensures a clean, consistent file handling pattern across all workflow tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from seer.core.files.models import WorkflowFileRef, is_file_ref, parse_file_ref
from seer.core.files.schemas import is_static_file_ref
from seer.logger import get_logger

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("seer.core.files.resolver")


class FileResolutionError(Exception):
    """Raised when file resolution fails."""


def _unwrap_file_wrapper(value: Any) -> Any:
    """
    Unwrap nested file reference wrappers from tool outputs.

    Some tools (e.g., google_drive_download_file) return wrapper objects:
        {"file": {...file_ref...}, "size_bytes": 123, "exported": true}

    When users reference the full output (${download}) as an attachment,
    we extract the nested file reference automatically.

    Args:
        value: Input value that might be a wrapper object.

    Returns:
        The unwrapped file reference if value is a wrapper, otherwise the original value.
    """
    if not isinstance(value, dict):
        return value

    # Check if this looks like a wrapper with a "file" key
    nested_file = value.get("file")
    if nested_file is None:
        return value

    # Only unwrap if the nested value is a valid file reference
    if is_file_ref(nested_file) or is_static_file_ref(nested_file):
        return nested_file

    return value


async def resolve_file_input(
    value: Any,
    context: Optional["WorkflowRuntimeContext"],
) -> Tuple[bytes, str, str]:
    """
    Resolve any file input to (bytes, mime_type, filename).

    This is the main entry point for tools that need to read file content.
    It handles both dynamic references (from parent nodes) and static
    references (from user storage).

    Args:
        value: File input value, either a WorkflowFileRef dict or static_file_ref dict.
        context: Workflow runtime context (required for file resolution).

    Returns:
        Tuple of (file_bytes, mime_type, filename).

    Raises:
        FileResolutionError: If the file cannot be resolved.
        ValueError: If the input format is invalid.
    """
    if value is None:
        raise ValueError("File input cannot be None")

    # Unwrap nested file references from tool output wrappers
    # (e.g., {"file": {...}, "size_bytes": 123} -> {...})
    value = _unwrap_file_wrapper(value)

    # Case 1: Dynamic WorkflowFileRef from parent node
    if is_file_ref(value):
        return await _resolve_workflow_file_ref(value, context)

    # Case 2: Static file reference from user storage
    if is_static_file_ref(value):
        return await _resolve_static_file_ref(value, context)

    # Case 3: Raw base64 string - REJECT
    if isinstance(value, str):
        raise ValueError(
            "Raw base64 input is not supported. Use a file reference from a parent node "
            "(e.g., ${download.file}) or a static file reference from user storage instead."
        )

    # Case 4: Unknown format
    raise ValueError(
        f"Invalid file input format: expected WorkflowFileRef or static_file_ref, "
        f"got {type(value).__name__}"
    )


async def _resolve_workflow_file_ref(
    value: dict[str, Any],
    context: Optional["WorkflowRuntimeContext"],
) -> Tuple[bytes, str, str]:
    """Resolve a WorkflowFileRef to file bytes."""
    if not context or not context.has_file_system:
        raise FileResolutionError(
            "WorkflowFileRef provided but workflow file system not available. "
            "This typically means the tool is being executed outside of a workflow context."
        )

    file_ref = parse_file_ref(value)
    content = await context.file_system.get_file_content(file_ref)
    logger.debug("Resolved WorkflowFileRef: file_id=%s size=%d", file_ref.file_id, len(content))

    return content, file_ref.mime_type, file_ref.filename


async def _resolve_static_file_ref(
    value: dict[str, Any],
    context: Optional["WorkflowRuntimeContext"],
) -> Tuple[bytes, str, str]:
    """Resolve a static_file_ref to file bytes."""
    if not context:
        raise FileResolutionError(
            "static_file_ref provided but no workflow context available. "
            "Static file references require workflow context for user verification."
        )

    file_id = value.get("file_id")
    if not file_id:
        raise ValueError("static_file_ref must have a file_id")

    if not context.has_file_system:
        raise FileResolutionError(
            "static_file_ref provided but workflow file system not available."
        )

    # Get file from user storage via the file system service
    content, file_ref = await context.file_system.get_file_by_id(file_id, context.user)
    logger.debug("Resolved static_file_ref: file_id=%s size=%d", file_id, len(content))

    return content, file_ref.mime_type, file_ref.filename


async def resolve_file_inputs(
    values: list[Any],
    context: Optional["WorkflowRuntimeContext"],
) -> list[Tuple[bytes, str, str]]:
    """
    Resolve multiple file inputs to (bytes, mime_type, filename) tuples.

    This is useful for tools that accept multiple files, like email attachments.

    Args:
        values: List of file input values.
        context: Workflow runtime context.

    Returns:
        List of (file_bytes, mime_type, filename) tuples.

    Raises:
        FileResolutionError: If any file cannot be resolved.
        ValueError: If any input format is invalid.
    """
    results = []
    for i, value in enumerate(values):
        try:
            result = await resolve_file_input(value, context)
            results.append(result)
        except (FileResolutionError, ValueError) as e:
            raise FileResolutionError(f"Failed to resolve file at index {i}: {e}") from e
    return results


def validate_file_input_format(value: Any) -> bool:
    """
    Check if a value is a valid file input format.

    This performs a quick structural check without actually resolving the file.
    Useful for validation before workflow execution.

    Args:
        value: Value to check.

    Returns:
        True if the value is a valid file input format.
    """
    # Unwrap nested file references from tool output wrappers
    value = _unwrap_file_wrapper(value)
    return is_file_ref(value) or is_static_file_ref(value)


async def get_file_metadata(
    value: Any,
    context: Optional["WorkflowRuntimeContext"],
) -> WorkflowFileRef:
    """
    Get file metadata without downloading the full content.

    For WorkflowFileRef, returns the ref directly.
    For static_file_ref, looks up the file metadata from the database.

    Args:
        value: File input value.
        context: Workflow runtime context.

    Returns:
        WorkflowFileRef with file metadata.

    Raises:
        FileResolutionError: If the file metadata cannot be retrieved.
        ValueError: If the input format is invalid.
    """
    # Unwrap nested file references from tool output wrappers
    value = _unwrap_file_wrapper(value)

    if is_file_ref(value):
        return parse_file_ref(value)

    if is_static_file_ref(value):
        if not context:
            raise FileResolutionError("Context required for static_file_ref metadata lookup")

        file_id = value.get("file_id")
        if not file_id:
            raise ValueError("static_file_ref must have a file_id")

        # Get metadata from database without downloading content
        return await context.file_system.get_file_metadata_by_id(file_id, context.user)

    raise ValueError(
        f"Invalid file input format: expected WorkflowFileRef or static_file_ref, "
        f"got {type(value).__name__}"
    )


__all__ = [
    "FileResolutionError",
    "resolve_file_input",
    "resolve_file_inputs",
    "validate_file_input_format",
    "get_file_metadata",
]
