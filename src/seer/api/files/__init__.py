"""
User-level file management API.

Provides endpoints for listing, downloading, uploading, and deleting files
across all workflow runs for a user.
"""

from seer.api.files.router import router

__all__ = ["router"]
