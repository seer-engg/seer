from seer.tools.base import register_tool
from seer.tools.google.gcalendar import (
    GoogleCalendarCreateEventTool,
    GoogleCalendarDeleteEventTool,
    GoogleCalendarGetEventTool,
    GoogleCalendarListEventsTool,
    GoogleCalendarUpdateEventTool,
)
from seer.tools.google.gdocs import (
    GoogleDocsCreateTool,
    GoogleDocsReadTool,
    GoogleDocsWriteTool,
)
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
    GoogleSheetsBatchReadTool,
    GoogleSheetsBatchUpdateSpreadsheetTool,
    GoogleSheetsBatchWriteTool,
    GoogleSheetsClearTool,
    GoogleSheetsCreateSpreadsheetTool,
    GoogleSheetsGetSpreadsheetTool,
    GoogleSheetsReadTool,
    GoogleSheetsWriteTool,
)
from seer.tools.google.youtube import (
    YouTubeGetChannelTool,
    YouTubeGetPlaylistItemsTool,
    YouTubeGetVideoTool,
    YouTubeListPlaylistsTool,
    YouTubeSearchTool,
    YouTubeUploadVideoTool,
)


def register_google_tools():
    register_tool(GmailReadTool())
    register_tool(GoogleSheetsAppendTool())
    register_tool(GoogleDriveListFilesTool())
    # Google Calendar
    register_tool(GoogleCalendarListEventsTool())
    register_tool(GoogleCalendarGetEventTool())
    register_tool(GoogleCalendarCreateEventTool())
    register_tool(GoogleCalendarUpdateEventTool())
    register_tool(GoogleCalendarDeleteEventTool())
    # Google Docs
    register_tool(GoogleDocsReadTool())
    register_tool(GoogleDocsWriteTool())
    register_tool(GoogleDocsCreateTool())
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
    # Google Sheets - write operations
    register_tool(GoogleSheetsWriteTool())
    register_tool(GoogleSheetsClearTool())
    register_tool(GoogleSheetsBatchWriteTool())
    register_tool(GoogleSheetsCreateSpreadsheetTool())
    register_tool(GoogleSheetsBatchUpdateSpreadsheetTool())
    # Google Sheets - read operations
    register_tool(GoogleSheetsReadTool())
    register_tool(GoogleSheetsBatchReadTool())
    register_tool(GoogleSheetsGetSpreadsheetTool())
    # YouTube - read operations
    register_tool(YouTubeSearchTool())
    register_tool(YouTubeGetVideoTool())
    register_tool(YouTubeGetChannelTool())
    register_tool(YouTubeListPlaylistsTool())
    register_tool(YouTubeGetPlaylistItemsTool())
    # YouTube - upload operations
    register_tool(YouTubeUploadVideoTool())


__all__ = [
    "register_google_tools",
]
