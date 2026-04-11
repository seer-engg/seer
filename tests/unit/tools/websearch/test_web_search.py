"""Unit tests for web search tool."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.websearch.web_search import WebSearchTool


def _exa_response(results=None, highlights=None):
    """Build a mock Exa API response."""
    exa_results = []
    for i, r in enumerate(results or []):
        item = {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
        }
        if highlights and i < len(highlights):
            item["highlights"] = highlights[i]
        if "text" in r:
            item["text"] = r["text"]
        exa_results.append(item)

    data = {"results": exa_results}
    return data


@pytest.mark.unit
class TestWebSearchToolMetadata:
    """Test WebSearchTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        return WebSearchTool()

    def test_tool_name(self, tool):
        assert tool.name == "web_search"

    def test_tool_description(self, tool):
        assert tool.description
        assert "search" in tool.description.lower()
        assert "exa" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        assert tool.integration_type == "websearch"

    def test_no_oauth_required(self, tool):
        assert tool.required_scopes == []

    def test_parameters_schema(self, tool):
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "query" in schema["required"]
        assert len(schema["required"]) == 1

        props = schema["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "search_depth" in props
        assert "include_answer" in props
        assert "include_raw_content" in props

        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"
        assert props["search_depth"]["type"] == "string"
        assert props["include_answer"]["type"] == "boolean"
        assert props["include_raw_content"]["type"] == "boolean"
        assert props["search_depth"]["enum"] == ["basic", "advanced"]
        assert props["max_results"]["default"] == 5
        assert props["search_depth"]["default"] == "basic"
        assert props["include_answer"]["default"] is True
        assert props["include_raw_content"]["default"] is False

    def test_output_schema(self, tool):
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "search_depth" in schema["properties"]
        assert "results" in schema["properties"]
        assert "result_count" in schema["properties"]
        assert "answer" in schema["properties"]

    def test_get_metadata(self, tool):
        metadata = tool.get_metadata()

        assert metadata["name"] == "web_search"
        assert "description" in metadata
        assert metadata["integration_type"] == "websearch"
        assert metadata["required_scopes"] == []
        assert "parameters" in metadata
        assert "output_schema" in metadata


@pytest.mark.unit
class TestWebSearchToolExecution:
    """Test WebSearchTool execution."""

    @pytest.fixture
    def tool(self):
        return WebSearchTool()

    @pytest.mark.asyncio
    async def test_error_when_api_key_not_configured(self, tool):
        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.exa_api_key = None

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"query": "test query"},
                )

            assert exc_info.value.status_code == 503
            assert "Exa API key not configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_successful_search_execution(self, tool):
        exa_resp = _exa_response(
            results=[
                {"title": "Test Result 1", "url": "https://example.com/1"},
                {"title": "Test Result 2", "url": "https://example.com/2"},
            ],
            highlights=[["This is the content snippet."], ["Another content snippet."]],
        )

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = exa_resp
        mock_response.raise_for_status = lambda: None

        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                access_token=None,
                arguments={"query": "test query"},
            )

            assert result["query"] == "test query"
            assert result["search_depth"] == "basic"
            assert result["result_count"] == 2
            assert len(result["results"]) == 2
            assert result["results"][0]["title"] == "Test Result 1"
            assert result["results"][0]["url"] == "https://example.com/1"
            assert result["results"][0]["content"] == "This is the content snippet."

            # Verify Exa API was called with POST
            call_kwargs = mock_client.post.call_args
            body = call_kwargs[1]["json"]
            assert body["query"] == "test query"
            assert body["numResults"] == 5

    @pytest.mark.asyncio
    async def test_advanced_search_uses_neural(self, tool):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = lambda: None

        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await tool.execute(
                access_token=None,
                arguments={"query": "test", "search_depth": "advanced"},
            )

            body = mock_client.post.call_args[1]["json"]
            assert body["type"] == "neural"

    @pytest.mark.asyncio
    async def test_basic_search_uses_auto(self, tool):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = lambda: None

        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            await tool.execute(
                access_token=None,
                arguments={"query": "test", "search_depth": "basic"},
            )

            body = mock_client.post.call_args[1]["json"]
            assert body["type"] == "auto"

    @pytest.mark.asyncio
    async def test_include_raw_content(self, tool):
        exa_resp = _exa_response(
            results=[{
                "title": "Test Result",
                "url": "https://example.com/1",
                "text": "Full page content here...",
            }],
            highlights=[["Snippet"]],
        )

        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = exa_resp
        mock_response.raise_for_status = lambda: None

        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                access_token=None,
                arguments={"query": "test", "include_raw_content": True},
            )

            assert result["results"][0]["raw_content"] == "Full page content here..."

    @pytest.mark.asyncio
    async def test_no_answer_when_disabled(self, tool):
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = _exa_response()
        mock_response.raise_for_status = lambda: None

        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = await tool.execute(
                access_token=None,
                arguments={"query": "test", "include_answer": False},
            )

            assert "answer" not in result

    @pytest.mark.asyncio
    async def test_exa_api_error(self, tool):
        with patch("seer.tools.websearch.web_search.config") as mock_config, \
             patch("seer.tools.websearch.exa_client.config") as mock_exa_config, \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_config.exa_api_key = "test-api-key"
            mock_exa_config.exa_api_key = "test-api-key"

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "429 Too Many Requests", request=httpx.Request("POST", "https://api.exa.ai/search"), response=httpx.Response(429)
            )
            mock_client_cls.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"query": "test"},
                )

            assert exc_info.value.status_code == 502
            assert "Web search failed" in str(exc_info.value.detail)


@pytest.mark.unit
class TestWebSearchToolRegistration:
    """Test that websearch tools can be registered."""

    def test_tools_can_be_imported(self):
        from seer.tools.websearch import (
            WebSearchTool,
            register_websearch_tools,
        )

        assert WebSearchTool is not None
        assert callable(register_websearch_tools)

    def test_tool_registration(self):
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.websearch import register_websearch_tools

        clear_registry()
        register_websearch_tools()

        tool = get_tool("web_search")
        assert tool is not None
        assert tool.name == "web_search"
        assert isinstance(tool, WebSearchTool)
