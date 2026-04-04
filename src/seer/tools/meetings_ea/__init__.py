"""meetingsEA tools — meeting bot management for recording and transcription."""
from seer.tools.base import register_tool
from seer.tools.meetings_ea.create_bot import MeetingsEACreateBotTool
from seer.tools.meetings_ea.get_bot import MeetingsEAGetBotTool
from seer.tools.meetings_ea.get_recording import MeetingsEAGetRecordingTool
from seer.tools.meetings_ea.get_transcript import MeetingsEAGetTranscriptTool
from seer.tools.meetings_ea.leave_meeting import MeetingsEALeaveMeetingTool
from seer.tools.meetings_ea.list_bots import MeetingsEAListBotsTool


def register_meetings_ea_tools() -> None:
    """Register all meetingsEA tools with the tool registry."""
    register_tool(MeetingsEACreateBotTool())
    register_tool(MeetingsEAGetBotTool())
    register_tool(MeetingsEAGetRecordingTool())
    register_tool(MeetingsEAGetTranscriptTool())
    register_tool(MeetingsEALeaveMeetingTool())
    register_tool(MeetingsEAListBotsTool())


__all__ = ["register_meetings_ea_tools"]
