"""
Unit tests for MCP discovery tools.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from seer.tools.discovery_shared import (
    tokenize as _tokenize,
    search_tools_intent as _search_tools_intent,
)


@pytest.mark.unit
class TestTokenize:
    """Tests for _tokenize helper function."""

    def test_tokenize_snake_case(self):
        """Test tokenizing snake_case strings."""
        result = _tokenize("gmail_create_draft")
        assert "gmail" in result
        assert "create" in result
        assert "draft" in result

    def test_tokenize_camel_case(self):
        """Test tokenizing camelCase strings."""
        result = _tokenize("createDraft")
        assert "create" in result
        assert "draft" in result

    def test_tokenize_mixed(self):
        """Test tokenizing mixed format strings."""
        result = _tokenize("gmail_createDraft_v2")
        assert "gmail" in result
        assert "create" in result
        assert "draft" in result
        assert "v2" in result

    def test_tokenize_empty(self):
        """Test tokenizing empty string."""
        assert _tokenize("") == set()
        assert _tokenize(None) == set()


@pytest.mark.unit
class TestSearchToolsIntent:
    """Tests for _search_tools_intent function."""

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_search_returns_matching_tools(self, mock_get_tools):
        """Test that search returns tools matching the query."""
        mock_get_tools.return_value = [
            {
                "name": "gmail_create_draft",
                "description": "Create a Gmail draft",
                "integration_type": "gmail",
            },
            {
                "name": "slack_send_message",
                "description": "Send a Slack message",
                "integration_type": "slack",
            },
        ]

        results = _search_tools_intent("create draft", top_k=5)

        assert len(results) > 0
        # Gmail tool should match "create draft"
        assert any("gmail" in r.get("name", "").lower() for r in results)

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_search_respects_integration_filter(self, mock_get_tools):
        """Test that integration filter boosts matching tools."""
        mock_get_tools.return_value = [
            {
                "name": "gmail_create_draft",
                "description": "Create a Gmail draft",
                "integration_type": "gmail",
            },
            {
                "name": "slack_post_message",
                "description": "Post a message to Slack",
                "integration_type": "slack",
            },
        ]

        results = _search_tools_intent("message", integration_filter="slack", top_k=5)

        assert len(results) > 0
        # Slack tool should rank higher due to filter
        if len(results) > 0:
            top_result = results[0]
            assert "slack" in top_result.get("name", "").lower() or \
                   "slack" in top_result.get("integration", "").lower()

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_search_returns_empty_for_no_match(self, mock_get_tools):
        """Test that search returns empty list when nothing matches."""
        mock_get_tools.return_value = [
            {
                "name": "gmail_create_draft",
                "description": "Create a Gmail draft",
                "integration_type": "gmail",
            },
        ]

        results = _search_tools_intent("xyz123nonexistent", top_k=5)

        assert results == []


@pytest.mark.unit
class TestSearchToolsMCP:
    """Tests for search_tools MCP tool - accessing underlying function."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_search_tools_returns_json(self, mock_get_tools):
        """Test that search_tools returns valid JSON."""
        mock_get_tools.return_value = [
            {
                "name": "gmail_create_draft",
                "description": "Create a Gmail draft",
                "integration_type": "gmail",
                "parameters": {},
            },
        ]

        # Access the underlying function via .fn attribute
        from seer.mcp.tools.discovery import search_tools
        result = await search_tools.fn("create draft")
        data = json.loads(result)

        assert "query" in data
        assert data["query"] == "create draft"

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_search_tools_handles_no_results(self, mock_get_tools):
        """Test that search_tools handles no results gracefully."""
        mock_get_tools.return_value = []

        from seer.mcp.tools.discovery import search_tools
        result = await search_tools.fn("nonexistent")
        data = json.loads(result)

        assert data["top_match"] is None
        assert "message" in data or "alternatives" in data


@pytest.mark.unit
class TestListToolsMCP:
    """Tests for list_tools MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_list_tools_returns_all_tools(self, mock_get_tools):
        """Test that list_tools returns all available tools."""
        mock_get_tools.return_value = [
            {
                "name": "tool1",
                "description": "Tool 1",
                "integration_type": "gmail",
                "parameters": {},
                "required_scopes": [],
            },
            {
                "name": "tool2",
                "description": "Tool 2",
                "integration_type": "slack",
                "parameters": {},
                "required_scopes": [],
            },
        ]

        from seer.mcp.tools.discovery import list_tools
        result = await list_tools.fn()
        data = json.loads(result)

        assert "tools" in data
        assert data["total"] == 2

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_list_tools_filters_by_integration(self, mock_get_tools):
        """Test that list_tools filters by integration type."""
        mock_get_tools.return_value = [
            {
                "name": "gmail_tool",
                "description": "Gmail tool",
                "integration_type": "gmail",
                "parameters": {},
                "required_scopes": [],
            },
        ]

        from seer.mcp.tools.discovery import list_tools
        result = await list_tools.fn(integration_type="gmail")
        data = json.loads(result)

        # Verify the filter was passed (shared module calls get_tools_by_integration twice)
        calls = mock_get_tools.call_args_list
        assert any(call.kwargs.get("integration_type") == "gmail" for call in calls)
        assert data["integration_filter"] == "gmail"


@pytest.mark.unit
class TestSearchTriggersMCP:
    """Tests for search_triggers MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.trigger_registry")
    async def test_search_triggers_returns_json(self, mock_registry):
        """Test that search_triggers returns valid JSON."""
        # Create mock trigger
        mock_trigger = MagicMock()
        mock_trigger.key = "poll.gmail.email_received"
        mock_trigger.title = "Gmail Email Received"
        mock_trigger.provider = "gmail"
        mock_trigger.mode = "polling"
        mock_trigger.description = "Triggered when new email arrives"
        mock_trigger.schemas = MagicMock()
        mock_trigger.schemas.config = None
        mock_trigger.schemas.event = None
        mock_trigger.meta = MagicMock()
        mock_trigger.meta.sample_event = None
        mock_trigger.meta.requires_connection = True

        mock_registry.all.return_value = [mock_trigger]

        from seer.mcp.tools.discovery import search_triggers
        result = await search_triggers.fn("gmail email")
        data = json.loads(result)

        assert "triggers" in data
        assert len(data["triggers"]) > 0
        assert data["triggers"][0]["key"] == "poll.gmail.email_received"


@pytest.mark.unit
class TestListTriggersMCP:
    """Tests for list_triggers MCP tool."""

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.trigger_registry")
    async def test_list_triggers_returns_all(self, mock_registry):
        """Test that list_triggers returns all triggers."""
        mock_trigger1 = MagicMock()
        mock_trigger1.key = "webhook.generic"
        mock_trigger1.title = "Generic Webhook"
        mock_trigger1.provider = "webhook"
        mock_trigger1.mode = "webhook"
        mock_trigger1.description = "Generic webhook trigger"
        mock_trigger1.schemas = MagicMock()
        mock_trigger1.schemas.event = None
        mock_trigger1.meta = MagicMock()
        mock_trigger1.meta.requires_connection = False
        mock_trigger1.meta.sample_event = None

        mock_trigger2 = MagicMock()
        mock_trigger2.key = "schedule.cron"
        mock_trigger2.title = "Cron Schedule"
        mock_trigger2.provider = "schedule"
        mock_trigger2.mode = "polling"
        mock_trigger2.description = "Cron-based schedule"
        mock_trigger2.schemas = MagicMock()
        mock_trigger2.schemas.event = None
        mock_trigger2.meta = MagicMock()
        mock_trigger2.meta.requires_connection = False
        mock_trigger2.meta.sample_event = None

        mock_registry.all.return_value = [mock_trigger1, mock_trigger2]

        from seer.mcp.tools.discovery import list_triggers
        result = await list_triggers.fn()
        data = json.loads(result)

        assert "triggers" in data
        assert data["total"] == 2
        assert "by_provider" in data
