from shared.tools.google.gmail import (
    GmailReadTool,
)
from shared.tools.google.gsheets import (
    GoogleSheetsAppendTool,
)
from shared.tools.google.gdrive import (
    GoogleDriveListFilesTool,
    GoogleDriveGetFileMetadataTool,
    GoogleDriveDownloadFileTool,
    GoogleDriveUploadFileTool,
    GoogleDriveCreateFolderTool,
    GoogleDriveUpdateFileTool,
    GoogleDriveDeleteFileTool,
    GoogleDriveCreatePermissionTool,
    GoogleDriveAboutGetTool,
)

from shared.tools.base import register_tool

def register_google_tools():
    register_tool(GmailReadTool())
    register_tool(GoogleSheetsAppendTool())
    register_tool(GoogleDriveListFilesTool())
    register_tool(GoogleDriveGetFileMetadataTool())
    register_tool(GoogleDriveDownloadFileTool())
    register_tool(GoogleDriveUploadFileTool())
    register_tool(GoogleDriveCreateFolderTool())
    register_tool(GoogleDriveUpdateFileTool())
    register_tool(GoogleDriveDeleteFileTool())
    register_tool(GoogleDriveCreatePermissionTool())
    register_tool(GoogleDriveAboutGetTool())

__all__ = [
    "register_google_tools",
]