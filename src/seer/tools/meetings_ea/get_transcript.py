"""meetingsEA tool: get meeting transcript."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.get_transcript")


class MeetingsEAGetTranscriptTool(MeetingsEAToolBase):
    """Get the transcript of a recorded meeting."""

    name = "meetings_ea_get_transcript"
    description = (
        "Get the transcript of a meetingsEA meeting, including speaker identification "
        "and timestamps."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bot_id": {
                    "type": "string",
                    "description": "The bot identifier returned by meetings_ea_create_bot.",
                },
            },
            "required": ["bot_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "utterances": {
                    "type": "array",
                    "description": "List of transcript entries",
                    "items": {
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "description": "Speaker name"},
                            "text": {"type": "string", "description": "Transcribed text"},
                            "start_ms": {"type": "integer", "description": "Start time in milliseconds"},
                            "duration_ms": {"type": "integer", "description": "Duration in milliseconds"},
                        },
                    },
                },
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, credentials

        bot_id: str = arguments["bot_id"]

        # Verify org ownership
        if context and context.organization_id is not None:
            bot = await self._api_request("GET", f"/api/v1/bots/{bot_id}")
            bot_metadata = bot.get("metadata") or {}
            bot_org_id = bot_metadata.get("seer_org_id")
            if bot_org_id and bot_org_id != str(context.organization_id):
                raise HTTPException(status_code=404, detail="Bot not found")

        result = await self._api_request("GET", f"/api/v1/bots/{bot_id}/transcript")

        # Normalize Attendee's utterance format
        raw_utterances = result if isinstance(result, list) else result.get("utterances", [])
        utterances: List[Dict[str, Any]] = []
        for u in raw_utterances:
            transcription = u.get("transcription") or {}
            text = transcription.get("transcript", "") if isinstance(transcription, dict) else str(transcription)
            utterances.append({
                "speaker": u.get("speaker_name", "Unknown"),
                "text": text,
                "start_ms": u.get("timestamp_ms", 0),
                "duration_ms": u.get("duration_ms", 0),
            })

        return {"utterances": utterances}


__all__ = ["MeetingsEAGetTranscriptTool"]
