"""Unit tests for semantic tool index."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from seer.tools.semantic_index import ToolSemanticIndex


def _make_index(tools, triggers, tool_embeddings, query_embedding):
    """Helper: build a ToolSemanticIndex with mocked embedding service."""
    mock_svc = MagicMock()
    mock_svc.embed_texts = AsyncMock(return_value=tool_embeddings)
    mock_svc.embed_text = AsyncMock(return_value=query_embedding)

    mock_trigger_reg = MagicMock()
    mock_trigger_reg.all.return_value = triggers

    mock_get_tools = MagicMock(return_value=tools)

    return mock_svc, mock_trigger_reg, mock_get_tools


@pytest.mark.unit
class TestToolSemanticIndex:
    @pytest.fixture
    def mock_tools(self):
        return [
            {"name": "http_request", "description": "Make HTTP requests to any URL", "integration_type": "http"},
            {"name": "gmail_send_email", "description": "Send an email using Gmail", "integration_type": "gmail"},
            {"name": "web_search", "description": "Search the web using Tavily", "integration_type": "websearch"},
        ]

    @pytest.fixture
    def mock_trigger(self):
        t = MagicMock()
        t.key = "schedule.cron"
        t.title = "Cron Schedule"
        t.description = "Time-based trigger"
        t.provider = "schedule"
        return t

    async def test_build_and_search(self, mock_tools, mock_trigger, monkeypatch):
        """Semantic search finds http_request for 'call external API'."""
        tool_embeddings = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]]
        query_embedding = [0.9, 0.1, 0]
        mock_svc, mock_reg, mock_get_tools = _make_index(mock_tools, [mock_trigger], tool_embeddings, query_embedding)

        index = ToolSemanticIndex()
        # Monkey-patch _build_index internals
        monkeypatch.setattr("seer.tools.registry.get_tools_by_integration", mock_get_tools)

        # Directly set up the index
        import numpy as np
        index._items = [
            {"name": t["name"], "description": t["description"], "integration_type": t["integration_type"], "item_type": "tool"}
            for t in mock_tools
        ] + [{"name": "schedule.cron", "description": "Time-based trigger", "integration_type": "schedule", "item_type": "trigger"}]
        index._embeddings = np.array(tool_embeddings, dtype=np.float32)
        index.is_initialized = True

        # Mock embedding service for search
        monkeypatch.setattr("seer.services.knowledge.embedding_service.EmbeddingService", lambda: mock_svc)

        results = await index.search("call external API", top_k=3)
        assert len(results) > 0
        assert results[0]["name"] == "http_request"

    async def test_item_type_filter(self, mock_tools, mock_trigger):
        """Filter by item_type excludes triggers."""
        import numpy as np

        index = ToolSemanticIndex()
        index._items = [
            {"name": "http_request", "description": "HTTP", "integration_type": "http", "item_type": "tool"},
            {"name": "schedule.cron", "description": "Cron", "integration_type": "schedule", "item_type": "trigger"},
        ]
        index._embeddings = np.array([[1, 0], [0.9, 0.1]], dtype=np.float32)
        index.is_initialized = True

        mock_svc = MagicMock()
        mock_svc.embed_text = AsyncMock(return_value=[0.95, 0.05])

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("seer.services.knowledge.embedding_service.EmbeddingService", lambda: mock_svc)
            results = await index.search("schedule", top_k=5, item_type="tool")
            assert all(r["item_type"] == "tool" for r in results)

    async def test_uninitialized_returns_empty(self):
        """Uninitialized index returns empty results."""
        index = ToolSemanticIndex()
        results = await index.search("anything")
        assert results == []
