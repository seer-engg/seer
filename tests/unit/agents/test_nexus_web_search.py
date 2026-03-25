"""
Tests for Nexus agent web_search tool.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.seer.agents.nexus.tools import web_search
from src.seer.agents.nexus.utils import get_workflow_tools

pytestmark = pytest.mark.unit


def test_web_search_tool_metadata():
    """Test that web_search tool has correct metadata."""
    assert web_search.name == "web_search"
    assert "Tavily" in web_search.description
    assert "web" in web_search.description.lower()


def test_web_search_not_in_workflow_tools():
    """Test that web_search is NOT included in get_workflow_tools() (removed for tool bloat reduction)."""
    tools = get_workflow_tools()
    tool_names = [t.name for t in tools]
    assert "web_search" not in tool_names


async def test_web_search_returns_error_without_api_key():
    """Test that web_search returns graceful error when API key is not configured."""
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = None

        result = await web_search.ainvoke({"query": "test query"})
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "not configured" in result_dict["error"]
        assert result_dict["query"] == "test query"
        assert "suggestion" in result_dict


async def test_web_search_successful_call():
    """Test that web_search correctly calls Tavily API and formats response."""
    mock_tavily_response = {
        "answer": "This is a test answer",
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com/test",
                "content": "Test content snippet",
                "score": 0.95,
            }
        ],
    }

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-api-key"

        # Patch the TavilyClient class in the tavily module
        with patch("tavily.TavilyClient") as mock_tavily_class:
            mock_client = MagicMock()
            mock_client.search.return_value = mock_tavily_response
            mock_tavily_class.return_value = mock_client

            result = await web_search.ainvoke({
                "query": "workflow automation",
                "max_results": 3,
            })
            result_dict = json.loads(result)

            # Verify Tavily was called correctly
            mock_tavily_class.assert_called_once_with(api_key="test-api-key")
            mock_client.search.assert_called_once()
            call_kwargs = mock_client.search.call_args.kwargs
            assert call_kwargs["query"] == "workflow automation"
            assert call_kwargs["max_results"] == 3

            # Verify response format
            assert result_dict["query"] == "workflow automation"
            assert result_dict["answer"] == "This is a test answer"
            assert len(result_dict["results"]) == 1
            assert result_dict["results"][0]["title"] == "Test Result"
            assert result_dict["result_count"] == 1


async def test_web_search_clamps_max_results():
    """Test that max_results is clamped to valid range (1-10)."""
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-api-key"

        with patch("tavily.TavilyClient") as mock_tavily_class:
            mock_client = MagicMock()
            mock_client.search.return_value = {"results": []}
            mock_tavily_class.return_value = mock_client

            # Test clamping upper bound
            await web_search.ainvoke({"query": "test", "max_results": 20})
            assert mock_client.search.call_args.kwargs["max_results"] == 10

            mock_client.reset_mock()

            # Test clamping lower bound
            await web_search.ainvoke({"query": "test", "max_results": 0})
            assert mock_client.search.call_args.kwargs["max_results"] == 1


async def test_web_search_handles_api_error():
    """Test that web_search handles API errors gracefully."""
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.tavily_api_key = "test-api-key"

        with patch("tavily.TavilyClient") as mock_tavily_class:
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("API rate limit exceeded")
            mock_tavily_class.return_value = mock_client

            result = await web_search.ainvoke({"query": "test query"})
            result_dict = json.loads(result)

            assert "error" in result_dict
            assert "rate limit" in result_dict["error"]
            assert result_dict["query"] == "test query"
