"""
Tests for Nexus agent web_search tool.
"""

import json
from unittest.mock import patch, AsyncMock

import httpx
import pytest

from src.seer.agents.nexus.tools import web_search
from src.seer.agents.nexus.utils import get_workflow_tools

pytestmark = pytest.mark.unit


def _brave_response(results=None, summary=None):
    """Build a mock Brave API response."""
    data = {"web": {"results": results or []}}
    if summary:
        data["summarizer"] = {"results": [{"summary": summary}]}
    return data


def test_web_search_tool_metadata():
    assert web_search.name == "web_search"
    assert "Brave" in web_search.description
    assert "web" in web_search.description.lower()


def test_web_search_not_in_workflow_tools():
    tools = get_workflow_tools()
    tool_names = [t.name for t in tools]
    assert "web_search" not in tool_names


async def test_web_search_returns_error_without_api_key():
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config:
        mock_config.brave_search_api_key = None

        result = await web_search.ainvoke({"query": "test query"})
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "not configured" in result_dict["error"]
        assert result_dict["query"] == "test query"
        assert "suggestion" in result_dict


async def test_web_search_successful_call():
    brave_resp = _brave_response(
        results=[{
            "title": "Test Result",
            "url": "https://example.com/test",
            "description": "Test content snippet",
        }],
        summary="This is a test answer",
    )

    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = brave_resp
    mock_response.raise_for_status = lambda: None

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config, \
         patch("seer.tools.websearch.brave_client.config") as mock_bc_config, \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_config.brave_search_api_key = "test-api-key"
        mock_bc_config.brave_search_api_key = "test-api-key"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = await web_search.ainvoke({
            "query": "workflow automation",
            "max_results": 3,
        })
        result_dict = json.loads(result)

        # Verify Brave API was called
        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["q"] == "workflow automation"
        assert call_kwargs[1]["params"]["count"] == 3

        # Verify response format
        assert result_dict["query"] == "workflow automation"
        assert result_dict["answer"] == "This is a test answer"
        assert len(result_dict["results"]) == 1
        assert result_dict["results"][0]["title"] == "Test Result"
        assert result_dict["result_count"] == 1


async def test_web_search_clamps_max_results():
    mock_response = AsyncMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = _brave_response()
    mock_response.raise_for_status = lambda: None

    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config, \
         patch("seer.tools.websearch.brave_client.config") as mock_bc_config, \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_config.brave_search_api_key = "test-api-key"
        mock_bc_config.brave_search_api_key = "test-api-key"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value = mock_client

        # Test clamping upper bound
        await web_search.ainvoke({"query": "test", "max_results": 20})
        assert mock_client.get.call_args[1]["params"]["count"] == 10

        mock_client.get.reset_mock()

        # Test clamping lower bound
        await web_search.ainvoke({"query": "test", "max_results": 0})
        assert mock_client.get.call_args[1]["params"]["count"] == 1


async def test_web_search_handles_api_error():
    with patch("src.seer.agents.nexus.tools.web_search.config") as mock_config, \
         patch("seer.tools.websearch.brave_client.config") as mock_bc_config, \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_config.brave_search_api_key = "test-api-key"
        mock_bc_config.brave_search_api_key = "test-api-key"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get.side_effect = Exception("API rate limit exceeded")
        mock_client_cls.return_value = mock_client

        result = await web_search.ainvoke({"query": "test query"})
        result_dict = json.loads(result)

        assert "error" in result_dict
        assert "rate limit" in result_dict["error"]
        assert result_dict["query"] == "test query"
