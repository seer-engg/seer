"""
Google Drive base classes shared across Drive tools.

Centralizes integration metadata and common Drive OAuth scopes to reduce duplication across tools.
"""
from seer.tools.google.base import GoogleAPIClient

# Shared integration metadata/constants
DRIVE_INTEGRATION_TYPE = "google_drive"
DRIVE_FILE_SCOPE = ["https://www.googleapis.com/auth/drive.file"]
DRIVE_METADATA_SCOPE = ["https://www.googleapis.com/auth/drive.metadata.readonly"]
DRIVE_READ_SCOPE = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveTool(GoogleAPIClient):
    """Base class for all Google Drive tools."""

    integration_type = DRIVE_INTEGRATION_TYPE


class GoogleDriveFileScopeTool(GoogleDriveTool):
    """Drive tool requiring read/write access."""

    required_scopes = DRIVE_FILE_SCOPE


class GoogleDriveMetadataScopeTool(GoogleDriveTool):
    """Drive tool requiring metadata read access."""

    required_scopes = DRIVE_METADATA_SCOPE


class GoogleDriveReadonlyScopeTool(GoogleDriveTool):
    """Drive tool requiring file read access."""

    required_scopes = DRIVE_READ_SCOPE
