"""
Unit tests for discovery tools (unified implementations).

Tests TF-IDF + substring matching in discovery_shared,
plus unified tool implementations.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from seer.tools.discovery_shared import (
    tokenize,
    search_tools_intent,
    _ToolIndex,
)


@pytest.mark.unit
class TestTokenize:
    def test_snake_case(self):
        assert "gmail" in tokenize("gmail_create_draft")
        assert "create" in tokenize("gmail_create_draft")
        assert "draft" in tokenize("gmail_create_draft")

    def test_camel_case(self):
        result = tokenize("createDraft")
        assert "create" in result
        assert "draft" in result

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize(None) == []


@pytest.mark.unit
class TestToolIndex:
    def test_exact_match_ranks_highest(self):
        catalog = [
            {"name": "http_request", "description": "Make HTTP requests to any URL"},
            {"name": "web_search", "description": "Search the web"},
            {"name": "gmail_send_email", "description": "Send an email"},
        ]
        index = _ToolIndex(catalog)
        results = index.search("http request")
        assert results[0] == 0  # http_request

    def test_substring_match_works(self):
        catalog = [
            {"name": "http_request", "description": "Make HTTP requests to any URL. Use for external APIs."},
            {"name": "gmail_send_email", "description": "Send an email using Gmail"},
        ]
        index = _ToolIndex(catalog)
        results = index.search("call an external API")
        assert results[0] == 0  # http_request (via "api" substring)

    def test_empty_query(self):
        catalog = [{"name": "test", "description": "test"}]
        index = _ToolIndex(catalog)
        assert index.search("") == []

    def test_empty_catalog(self):
        index = _ToolIndex([])
        assert index.search("anything") == []

    def test_no_match(self):
        catalog = [{"name": "gmail_send", "description": "Send email"}]
        index = _ToolIndex(catalog)
        assert index.search("xyznonexistent") == []


@pytest.mark.unit
class TestSearchToolsIntent:
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_returns_matching_tools(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_create_draft", "description": "Create a Gmail draft", "integration_type": "gmail"},
            {"name": "slack_send_message", "description": "Send a Slack message", "integration_type": "slack"},
        ]
        results = search_tools_intent("create draft", top_k=5)
        assert len(results) > 0
        assert results[0]["name"] == "gmail_create_draft"

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_filters_by_integration(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_create_draft", "description": "Create a Gmail draft", "integration_type": "gmail"},
            {"name": "slack_post_message", "description": "Post a Slack message", "integration_type": "slack"},
        ]
        results = search_tools_intent("message", integration_filter="slack", top_k=5)
        assert len(results) == 1
        assert results[0]["name"] == "slack_post_message"

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_returns_empty_for_no_tools(self, mock_get_tools):
        mock_get_tools.return_value = []
        results = search_tools_intent("anything")
        assert results == []

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_results_have_confidence_score(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "http_request", "description": "Make HTTP requests", "integration_type": "http"},
        ]
        results = search_tools_intent("http request")
        assert len(results) == 1
        assert "confidence_score" in results[0]

    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    def test_results_are_compact(self, mock_get_tools):
        """Search results should not include parameter schemas."""
        mock_get_tools.return_value = [
            {"name": "http_request", "description": "Make HTTP requests",
             "integration_type": "http", "parameters": {"type": "object"}, "required_scopes": []},
        ]
        results = search_tools_intent("http")
        assert "parameters" not in results[0]
        assert "required_scopes" not in results[0]


@pytest.mark.unit
class TestSearchToolsUnified:
    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.async_search_tools_intent")
    async def test_returns_json(self, mock_search):
        mock_search.return_value = [
            {"name": "gmail_create_draft", "description": "Create a Gmail draft",
             "integration_type": "gmail", "parameters": {}, "confidence_score": 0.95},
        ]
        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("create draft")
        data = json.loads(result)
        assert data["query"] == "create draft"

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.async_search_tools_intent")
    async def test_handles_no_results(self, mock_search):
        mock_search.return_value = []
        from seer.tools.unified_tools import search_tools_impl
        result = await search_tools_impl("nonexistent")
        data = json.loads(result)
        assert data["top_match"] is None


@pytest.mark.unit
class TestListToolsUnified:
    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_returns_all_tools(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "tool1", "description": "Tool 1", "integration_type": "gmail", "parameters": {}, "required_scopes": []},
            {"name": "tool2", "description": "Tool 2", "integration_type": "slack", "parameters": {}, "required_scopes": []},
        ]
        from seer.tools.unified_tools import list_tools_impl
        data = json.loads(await list_tools_impl())
        assert data["total"] == 2

    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.get_tools_by_integration")
    async def test_filters_by_integration(self, mock_get_tools):
        mock_get_tools.return_value = [
            {"name": "gmail_tool", "description": "Gmail", "integration_type": "gmail", "parameters": {}, "required_scopes": []},
        ]
        from seer.tools.unified_tools import list_tools_impl
        data = json.loads(await list_tools_impl(integration_type="gmail"))
        assert data["integration_filter"] == "gmail"


@pytest.mark.unit
class TestSearchTriggersUnified:
    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.async_search_triggers_intent")
    async def test_returns_json(self, mock_search):
        mock_search.return_value = [
            {
                "key": "poll.gmail.email_received",
                "title": "Gmail Email Received",
                "provider": "gmail",
                "mode": "polling",
                "description": "Triggered when new email arrives",
                "config_schema": None,
                "event_schema": None,
                "sample_event": None,
                "requires_connection": True,
                "confidence_score": 0.92,
            },
        ]

        from seer.tools.unified_tools import search_triggers_impl
        data = json.loads(await search_triggers_impl("gmail email"))
        assert len(data["triggers"]) > 0
        assert data["triggers"][0]["key"] == "poll.gmail.email_received"


@pytest.mark.unit
class TestListTriggersUnified:
    @pytest.mark.asyncio
    @patch("seer.tools.discovery_shared.trigger_registry")
    async def test_returns_all(self, mock_registry):
        t1 = MagicMock()
        t1.key = "webhook.generic"
        t1.title = "Generic Webhook"
        t1.provider = "webhook"
        t1.mode = "webhook"
        t1.description = "Generic webhook trigger"
        t1.schemas = MagicMock()
        t1.schemas.event = None
        t1.meta = MagicMock()
        t1.meta.requires_connection = False
        t1.meta.sample_event = None

        t2 = MagicMock()
        t2.key = "schedule.cron"
        t2.title = "Cron Schedule"
        t2.provider = "schedule"
        t2.mode = "polling"
        t2.description = "Cron-based schedule"
        t2.schemas = MagicMock()
        t2.schemas.event = None
        t2.meta = MagicMock()
        t2.meta.requires_connection = False
        t2.meta.sample_event = None

        mock_registry.all.return_value = [t1, t2]

        from seer.tools.unified_tools import list_triggers_impl
        data = json.loads(await list_triggers_impl())
        assert data["total"] == 2
        assert "by_provider" in data
