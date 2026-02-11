"""
Pydantic models for user-level file management API.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class UserFileListItem(BaseModel):
    """File metadata for a user's file."""

    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    size_human: str
    run_id: Optional[str] = None  # Null for user-uploaded files
    workflow_id: Optional[str] = None
    workflow_name: Optional[str] = None
    source_node_id: Optional[str] = None
    source_tool: Optional[str] = None
    created_at: datetime


class UserFileListResponse(BaseModel):
    """Response containing list of files for a user."""

    files: List[UserFileListItem]
    total_count: int
    total_size_bytes: int
    next_cursor: Optional[str] = None


class UserFileResponse(BaseModel):
    """Response containing single file metadata."""

    file: UserFileListItem


class UserFileDownloadResponse(BaseModel):
    """Response containing presigned download URL."""

    file_id: str
    filename: str
    download_url: str
    expires_in_seconds: int


class UserFileDeleteResponse(BaseModel):
    """Response confirming file deletion."""

    file_id: str
    deleted: bool


class MimeTypeStats(BaseModel):
    """Statistics for a MIME type."""

    mime_type: str
    file_count: int
    total_size_bytes: int
    total_size_human: str


class ToolStats(BaseModel):
    """Statistics for a source tool."""

    source_tool: str
    file_count: int
    total_size_bytes: int


class UserStorageStatsResponse(BaseModel):
    """Response containing storage statistics for a user."""

    total_files: int
    total_size_bytes: int
    total_size_human: str
    files_by_mime_type: List[MimeTypeStats]
    files_by_tool: List[ToolStats]
    oldest_file_date: Optional[datetime] = None
    newest_file_date: Optional[datetime] = None


class BulkDeleteFilesRequest(BaseModel):
    """Request to delete multiple files."""

    file_ids: List[str] = Field(..., min_length=1, max_length=100)


class BulkDeleteResult(BaseModel):
    """Result of deleting a single file in bulk operation."""

    file_id: str
    deleted: bool
    error: Optional[str] = None


class BulkDeleteFilesResponse(BaseModel):
    """Response from bulk delete operation."""

    results: List[BulkDeleteResult]
    deleted_count: int
    failed_count: int
    total_size_freed_bytes: int


class FileSearchResponse(BaseModel):
    """Response from file search."""

    query: str
    results: List[UserFileListItem]
    total_matches: int


class UserFileUploadResponse(BaseModel):
    """Response after uploading a file."""

    file_id: str
    filename: str
    mime_type: str
    size_bytes: int
    size_human: str
    created_at: datetime


__all__ = [
    "UserFileListItem",
    "UserFileListResponse",
    "UserFileResponse",
    "UserFileDownloadResponse",
    "UserFileDeleteResponse",
    "MimeTypeStats",
    "ToolStats",
    "UserStorageStatsResponse",
    "BulkDeleteFilesRequest",
    "BulkDeleteResult",
    "BulkDeleteFilesResponse",
    "FileSearchResponse",
    "UserFileUploadResponse",
]
