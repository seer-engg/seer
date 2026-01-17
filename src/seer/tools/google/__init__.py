from seer.tools.base import register_tool
from seer.tools.google.gdrive import (
    GoogleDriveAboutGetTool,
    GoogleDriveCreateFolderTool,
    GoogleDriveCreatePermissionTool,
    GoogleDriveDeleteFileTool,
    GoogleDriveDownloadFileTool,
    GoogleDriveGetFileMetadataTool,
    GoogleDriveListFilesTool,
    GoogleDriveUpdateFileTool,
    GoogleDriveUploadFileTool,
)
from seer.tools.google.gmail import (
    GmailCreateDraftTool,
    GmailCreateLabelTool,
    GmailDeleteDraftTool,
    GmailDeleteLabelTool,
    GmailDeleteMessageTool,
    GmailGetAttachmentTool,
    GmailGetDraftTool,
    GmailGetMessageTool,
    GmailGetThreadTool,
    GmailListDraftsTool,
    GmailListLabelsTool,
    GmailListThreadsTool,
    GmailModifyMessageLabelsTool,
    GmailReadTool,
    GmailSendDraftTool,
    GmailSendEmailTool,
    GmailTrashMessageTool,
)
from seer.tools.google.gsheets import (
    GoogleSheetsAppendTool,
    GoogleSheetsWriteTool,
)


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
    register_tool(GmailSendEmailTool())
    register_tool(GmailGetMessageTool())
    register_tool(GmailModifyMessageLabelsTool())
    register_tool(GmailTrashMessageTool())
    register_tool(GmailDeleteMessageTool())
    register_tool(GmailListThreadsTool())
    register_tool(GmailGetThreadTool())
    register_tool(GmailCreateDraftTool())
    register_tool(GmailListDraftsTool())
    register_tool(GmailGetDraftTool())
    register_tool(GmailSendDraftTool())
    register_tool(GmailDeleteDraftTool())
    register_tool(GmailListLabelsTool())
    register_tool(GmailCreateLabelTool())
    register_tool(GmailDeleteLabelTool())
    register_tool(GmailGetAttachmentTool())
    register_tool(GoogleSheetsWriteTool())


__all__ = [
    "register_google_tools",
]
