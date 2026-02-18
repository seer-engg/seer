"""Unit tests for Slack channel tools."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.slack.channels import (
    SlackListChannelsTool,
    SlackGetChannelHistoryTool,
    SlackJoinChannelTool,
)


@pytest.mark.unit
class TestSlackListChannelsTool:
    """Tests for SlackListChannelsTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = SlackListChannelsTool()
        assert tool.name == "slack_list_channels"
        assert tool.integration_type == "slack"
        assert "channels:read" in tool.required_scopes
        assert "groups:read" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = SlackListChannelsTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "workspace_id" in schema["properties"]
        assert "types" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "workspace_id" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has expected fields."""
        tool = SlackListChannelsTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "ok" in schema["properties"]
        assert "channels" in schema["properties"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = SlackListChannelsTool()
        pickers = tool.get_resource_pickers()

        assert "workspace_id" in pickers
        assert pickers["workspace_id"]["resource_type"] == "workspace"

    @pytest.mark.asyncio
    async def test_execute_missing_workspace_id(self):
        """Test execute raises error when workspace_id is missing."""
        tool = SlackListChannelsTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {})

        assert exc_info.value.status_code == 400
        assert "workspace_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful channel listing."""
        tool = SlackListChannelsTool()

        mock_response = {
            "ok": True,
            "channels": [
                {"id": "C123", "name": "general", "is_private": False},
                {"id": "C456", "name": "random", "is_private": False},
            ],
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute("test_token", {"workspace_id": "T123"})

            assert result["ok"] is True
            assert len(result["channels"]) == 2
            mock_request.assert_called_once()

            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "conversations.list"


@pytest.mark.unit
class TestSlackGetChannelHistoryTool:
    """Tests for SlackGetChannelHistoryTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = SlackGetChannelHistoryTool()
        assert tool.name == "slack_get_channel_history"
        assert tool.integration_type == "slack"
        assert "channels:history" in tool.required_scopes

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = SlackGetChannelHistoryTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "workspace_id" in schema["properties"]
        assert "channel_id" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "workspace_id" in schema["required"]
        assert "channel_id" in schema["required"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = SlackGetChannelHistoryTool()
        pickers = tool.get_resource_pickers()

        assert "workspace_id" in pickers
        assert "channel_id" in pickers
        assert pickers["channel_id"]["depends_on"] == "workspace_id"

    @pytest.mark.asyncio
    async def test_execute_missing_workspace_id(self):
        """Test execute raises error when workspace_id is missing."""
        tool = SlackGetChannelHistoryTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"channel_id": "C123"})

        assert exc_info.value.status_code == 400
        assert "workspace_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_missing_channel_id(self):
        """Test execute raises error when channel_id is missing."""
        tool = SlackGetChannelHistoryTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"workspace_id": "T123"})

        assert exc_info.value.status_code == 400
        assert "channel_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful channel history retrieval."""
        tool = SlackGetChannelHistoryTool()

        mock_response = {
            "ok": True,
            "messages": [
                {"type": "message", "user": "U123", "text": "Hello", "ts": "1234567890.123456"},
            ],
            "has_more": False,
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {"workspace_id": "T123", "channel_id": "C123"},
            )

            assert result["ok"] is True
            assert len(result["messages"]) == 1
            mock_request.assert_called_once()

            call_args = mock_request.call_args
            assert call_args[0][0] == "GET"
            assert call_args[0][1] == "conversations.history"


@pytest.mark.unit
class TestSlackJoinChannelTool:
    """Tests for SlackJoinChannelTool."""

    def test_tool_attributes(self):
        """Test tool has correct attributes."""
        tool = SlackJoinChannelTool()
        assert tool.name == "slack_join_channel"
        assert tool.integration_type == "slack"
        assert "channels:join" in tool.required_scopes

    def test_description_mentions_public_channels(self):
        """Test description clarifies this is for public channels."""
        tool = SlackJoinChannelTool()
        assert "public" in tool.description.lower()
        assert "private" in tool.description.lower()

    def test_get_parameters_schema(self):
        """Test parameter schema includes required fields."""
        tool = SlackJoinChannelTool()
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "workspace_id" in schema["properties"]
        assert "channel_id" in schema["properties"]
        assert "workspace_id" in schema["required"]
        assert "channel_id" in schema["required"]

    def test_get_output_schema(self):
        """Test output schema has expected fields."""
        tool = SlackJoinChannelTool()
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "ok" in schema["properties"]
        assert "channel" in schema["properties"]

    def test_get_resource_pickers(self):
        """Test resource pickers configuration."""
        tool = SlackJoinChannelTool()
        pickers = tool.get_resource_pickers()

        assert "workspace_id" in pickers
        assert "channel_id" in pickers
        assert pickers["channel_id"]["depends_on"] == "workspace_id"
        assert pickers["workspace_id"]["resource_type"] == "workspace"
        assert pickers["channel_id"]["resource_type"] == "channel"

    @pytest.mark.asyncio
    async def test_execute_missing_workspace_id(self):
        """Test execute raises error when workspace_id is missing."""
        tool = SlackJoinChannelTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"channel_id": "C123"})

        assert exc_info.value.status_code == 400
        assert "workspace_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_missing_channel_id(self):
        """Test execute raises error when channel_id is missing."""
        tool = SlackJoinChannelTool()

        with pytest.raises(HTTPException) as exc_info:
            await tool.execute("test_token", {"workspace_id": "T123"})

        assert exc_info.value.status_code == 400
        assert "channel_id" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful channel join."""
        tool = SlackJoinChannelTool()

        mock_response = {
            "ok": True,
            "channel": {
                "id": "C123",
                "name": "general",
                "is_channel": True,
                "is_member": True,
            },
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {"workspace_id": "T123", "channel_id": "C123"},
            )

            assert result["ok"] is True
            assert result["channel"]["id"] == "C123"
            assert result["channel"]["is_member"] is True
            mock_request.assert_called_once()

            # Verify correct API call
            call_args = mock_request.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "conversations.join"

            # Verify request body
            json_body = call_args[1].get("json_body")
            assert json_body is not None
            assert json_body["channel"] == "C123"

    @pytest.mark.asyncio
    async def test_execute_already_in_channel(self):
        """Test joining a channel the bot is already in (should succeed)."""
        tool = SlackJoinChannelTool()

        # Slack returns success even if already in channel
        mock_response = {
            "ok": True,
            "channel": {
                "id": "C123",
                "name": "general",
                "is_channel": True,
                "is_member": True,
            },
        }

        with patch.object(tool, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await tool.execute(
                "test_token",
                {"workspace_id": "T123", "channel_id": "C123"},
            )

            assert result["ok"] is True
            assert result["channel"]["is_member"] is True
