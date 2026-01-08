"""Backwards-compatible entrypoint for Google Drive tools.

This module re-exports the smaller, split implementations:
- shared.tools.google._common
- shared.tools.google.drive_read_tools
- shared.tools.google.drive_write_tools

Prefer importing directly from those modules.
"""
from shared.tools.google.drive_read_tools import (
    GoogleDriveListFilesTool,
    GoogleDriveGetFileMetadataTool,
    GoogleDriveDownloadFileTool,
    GoogleDriveAboutGetTool,
)
from shared.tools.google.drive_write_tools import (
    GoogleDriveUploadFileTool,
    GoogleDriveCreateFolderTool,
    GoogleDriveUpdateFileTool,
    GoogleDriveDeleteFileTool,
    GoogleDriveCreatePermissionTool,
)

__all__ = [
    "GoogleDriveListFilesTool",
    "GoogleDriveGetFileMetadataTool",
    "GoogleDriveDownloadFileTool",
    "GoogleDriveAboutGetTool",
    "GoogleDriveUploadFileTool",
    "GoogleDriveCreateFolderTool",
    "GoogleDriveUpdateFileTool",
    "GoogleDriveDeleteFileTool",
    "GoogleDriveCreatePermissionTool",
]
