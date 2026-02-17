"""
Workflow File System - handles file storage and references for workflow execution.

This module provides a file storage abstraction that allows workflow tools to:
1. Store files in S3-compatible storage (AWS S3 or Cloudflare R2)
2. Return lightweight file references instead of raw data
3. Resolve file references on-demand when consuming tools need the data

Usage:
    from seer.core.files import WorkflowFileSystem, WorkflowFileRef

    # Store a file
    fs = WorkflowFileSystem.instance()
    file_ref = await fs.store_file(run_id, "document.pdf", data, "application/pdf")

    # Retrieve file content
    content = await fs.get_file_content(file_ref)

    # Resolve file input (handles both WorkflowFileRef and static_file_ref)
    from seer.core.files.resolver import resolve_file_input
    content, mime_type, filename = await resolve_file_input(file_input, context)
"""

from seer.core.files.models import WorkflowFileRef, is_file_ref, parse_file_ref
from seer.core.files.resolver import FileResolutionError, resolve_file_input, resolve_file_inputs
from seer.core.files.schemas import (
    FILE_INPUT_SCHEMA,
    FILE_OUTPUT_SCHEMA,
    STATIC_FILE_REF_TYPE,
    is_static_file_ref,
)
from seer.core.files.service import WorkflowFileSystem

__all__ = [
    "WorkflowFileRef",
    "WorkflowFileSystem",
    "is_file_ref",
    "parse_file_ref",
    "FileResolutionError",
    "resolve_file_input",
    "resolve_file_inputs",
    "FILE_INPUT_SCHEMA",
    "FILE_OUTPUT_SCHEMA",
    "STATIC_FILE_REF_TYPE",
    "is_static_file_ref",
]
