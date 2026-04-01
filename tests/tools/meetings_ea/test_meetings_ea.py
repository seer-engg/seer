"""Tests for meetingsEA tools."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.meetings_ea.base import MeetingsEAToolBase
from seer.tools.meetings_ea.create_bot import MeetingsEACreateBotTool
from seer.tools.meetings_ea.get_bot import MeetingsEAGetBotTool
from seer.tools.meetings_ea.get_recording import MeetingsEAGetRecordingTool
from seer.tools.meetings_ea.get_transcript import MeetingsEAGetTranscriptTool
from seer.tools.meetings_ea.leave_meeting import MeetingsEALeaveMeetingTool
from seer.tools.meetings_ea.list_bots import MeetingsEAListBotsTool


def _make_context(org_id: int = 1) -> MagicMock:
    ctx = MagicMock()
    ctx.organization_id = org_id
    return ctx


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Create a mock httpx Response."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("GET", "http://test"),
    )
    return response


class TestToolRegistration:
    """Verify all 6 tools register correctly."""

    def test_all_tools_registered(self):
        from seer.tools.base import list_tools
        from seer.tools.meetings_ea import register_meetings_ea_tools

        register_meetings_ea_tools()

        tools = list_tools()
        meetings_ea_tools = [t for t in tools if t.integration_type == "meetings_ea"]
        assert len(meetings_ea_tools) == 6

        names = {t.name for t in meetings_ea_tools}
        assert names == {
            "meetings_ea_create_bot",
            "meetings_ea_get_bot",
            "meetings_ea_list_bots",
            "meetings_ea_get_recording",
            "meetings_ea_get_transcript",
            "meetings_ea_leave_meeting",
        }

    def test_tool_metadata(self):
        tool = MeetingsEACreateBotTool()
        assert tool.provider is None
        assert tool.integration_type == "meetings_ea"
        assert tool.required_scopes == []


class TestMeetingsEACreateBot:

    @pytest.fixture
    def tool(self):
        return MeetingsEACreateBotTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_create_bot_success(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = "https://seer.test"

        mock_response = _mock_response({
            "id": "bot_123",
            "state": "joining",
            "meeting_url": "https://meet.google.com/abc",
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=42)
        result = await tool.execute(
            None,
            {"meeting_url": "https://meet.google.com/abc"},
            context=ctx,
        )

        assert result["bot_id"] == "bot_123"
        assert result["state"] == "joining"

        # Verify the request was made with correct auth and body
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"
        assert "/api/v1/bots" in call_args[0][1]
        assert call_args[1]["headers"]["Authorization"] == "Token test-key"

        body = call_args[1]["json"]
        assert body["meeting_url"] == "https://meet.google.com/abc"
        assert body["bot_name"] == "meetingsEA"
        assert body["metadata"]["seer_org_id"] == "42"
        assert body["webhooks"][0]["url"] == "https://seer.test/api/v1/webhooks/meetings_ea"

    @patch("seer.tools.meetings_ea.base.config")
    async def test_create_bot_missing_config(self, mock_config, tool):
        mock_config.meetings_ea_api_key = None
        mock_config.meetings_ea_base_url = None

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"meeting_url": "https://meet.google.com/abc"})
        assert exc_info.value.status_code == 503


class TestMeetingsEAGetBot:

    @pytest.fixture
    def tool(self):
        return MeetingsEAGetBotTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_get_bot_success(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        mock_response = _mock_response({
            "id": "bot_123",
            "state": "joined_recording",
            "meeting_url": "https://meet.google.com/abc",
            "metadata": {"seer_org_id": "1"},
            "events": [{"type": "state_change", "state": "joining"}],
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        result = await tool.execute(None, {"bot_id": "bot_123"}, context=ctx)

        assert result["bot_id"] == "bot_123"
        assert result["state"] == "joined_recording"

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_get_bot_wrong_org(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        mock_response = _mock_response({
            "id": "bot_123",
            "state": "ended",
            "meeting_url": "https://meet.google.com/abc",
            "metadata": {"seer_org_id": "999"},
            "events": [],
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"bot_id": "bot_123"}, context=ctx)
        assert exc_info.value.status_code == 404


class TestMeetingsEAGetTranscript:

    @pytest.fixture
    def tool(self):
        return MeetingsEAGetTranscriptTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_get_transcript_normalizes_output(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        # First call: get bot (for org check)
        bot_response = _mock_response({
            "id": "bot_123",
            "metadata": {"seer_org_id": "1"},
        })
        # Second call: get transcript
        transcript_response = _mock_response([
            {
                "speaker_name": "Alice",
                "timestamp_ms": 1000,
                "duration_ms": 5000,
                "transcription": {"transcript": "Hello everyone"},
            },
            {
                "speaker_name": "Bob",
                "timestamp_ms": 6000,
                "duration_ms": 3000,
                "transcription": {"transcript": "Hi Alice"},
            },
        ])

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=[bot_response, transcript_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        result = await tool.execute(None, {"bot_id": "bot_123"}, context=ctx)

        assert len(result["utterances"]) == 2
        assert result["utterances"][0]["speaker"] == "Alice"
        assert result["utterances"][0]["text"] == "Hello everyone"
        assert result["utterances"][0]["start_ms"] == 1000
        assert result["utterances"][1]["speaker"] == "Bob"


class TestMeetingsEAListBots:

    @pytest.fixture
    def tool(self):
        return MeetingsEAListBotsTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_list_bots_filters_by_org(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        mock_response = _mock_response({
            "results": [
                {"id": "bot_1", "state": "ended", "meeting_url": "url1", "metadata": {"seer_org_id": "1"}},
                {"id": "bot_2", "state": "ended", "meeting_url": "url2", "metadata": {"seer_org_id": "999"}},
                {"id": "bot_3", "state": "joining", "meeting_url": "url3", "metadata": {"seer_org_id": "1"}},
            ]
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        result = await tool.execute(None, {}, context=ctx)

        assert len(result["bots"]) == 2
        assert {b["bot_id"] for b in result["bots"]} == {"bot_1", "bot_3"}

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_list_bots_with_filters(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        mock_response = _mock_response({"results": []})

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        await tool.execute(
            None,
            {"meeting_url": "https://meet.google.com/abc", "status": "ended"},
            context=ctx,
        )

        call_args = mock_client.request.call_args
        assert call_args[1]["params"]["meeting_url"] == "https://meet.google.com/abc"
        assert call_args[1]["params"]["states"] == "ended"


class TestMeetingsEALeaveMeeting:

    @pytest.fixture
    def tool(self):
        return MeetingsEALeaveMeetingTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_leave_meeting_success(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        # First call: get bot (org check), second: leave
        bot_response = _mock_response({
            "id": "bot_123",
            "metadata": {"seer_org_id": "1"},
        })
        leave_response = _mock_response({
            "id": "bot_123",
            "state": "leaving",
        })

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=[bot_response, leave_response])
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        ctx = _make_context(org_id=1)
        result = await tool.execute(None, {"bot_id": "bot_123"}, context=ctx)

        assert result["bot_id"] == "bot_123"
        assert result["state"] == "leaving"


class TestMeetingsEAApiErrorHandling:

    @pytest.fixture
    def tool(self):
        return MeetingsEAGetBotTool()

    @patch("seer.tools.meetings_ea.base.config")
    @patch("seer.tools.meetings_ea.base.httpx.AsyncClient")
    async def test_api_error_returns_502(self, mock_client_cls, mock_config, tool):
        mock_config.meetings_ea_api_key = "test-key"
        mock_config.meetings_ea_base_url = "https://attendee.test"
        mock_config.webhook_base_url = None

        error_response = httpx.Response(
            status_code=500,
            json={"detail": "Internal server error"},
            request=httpx.Request("GET", "http://test"),
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=error_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute(None, {"bot_id": "bot_123"})
        assert exc_info.value.status_code == 502
        assert "meetingsEA" in str(exc_info.value.detail)
