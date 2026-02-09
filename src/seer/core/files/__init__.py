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
"""

from seer.core.files.models import WorkflowFileRef
from seer.core.files.service import WorkflowFileSystem

__all__ = ["WorkflowFileRef", "WorkflowFileSystem"]
