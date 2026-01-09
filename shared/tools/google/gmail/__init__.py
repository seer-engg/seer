"""
Gmail tools - reading, sending, managing emails.

Backward-compatible imports for existing code.
"""

# Export all tools for backward compatibility
from shared.tools.google.gmail.read import (
    GmailGetAttachmentTool,
    GmailGetMessageTool,
    GmailGetThreadTool,
    GmailListThreadsTool,
    GmailReadTool,
)
from shared.tools.google.gmail.send import GmailSendEmailTool
from shared.tools.google.gmail.manage import (
    GmailCreateLabelTool,
    GmailDeleteLabelTool,
    GmailDeleteMessageTool,
    GmailListLabelsTool,
    GmailModifyMessageLabelsTool,
    GmailTrashMessageTool,
)
from shared.tools.google.gmail.drafts import (
    GmailCreateDraftTool,
    GmailDeleteDraftTool,
    GmailGetDraftTool,
    GmailListDraftsTool,
    GmailSendDraftTool,
)

__all__ = [
    # Read operations
    "GmailReadTool",
    "GmailGetMessageTool",
    "GmailListThreadsTool",
    "GmailGetThreadTool",
    "GmailGetAttachmentTool",
    # Send operations
    "GmailSendEmailTool",
    # Manage operations
    "GmailModifyMessageLabelsTool",
    "GmailTrashMessageTool",
    "GmailDeleteMessageTool",
    "GmailListLabelsTool",
    "GmailCreateLabelTool",
    "GmailDeleteLabelTool",
    # Draft operations
    "GmailCreateDraftTool",
    "GmailListDraftsTool",
    "GmailGetDraftTool",
    "GmailSendDraftTool",
    "GmailDeleteDraftTool",
]
