"""Unit tests for web search tool."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.websearch.web_search import WebSearchTool


@pytest.mark.unit
class TestWebSearchToolMetadata:
    """Test WebSearchTool metadata and schema."""

    @pytest.fixture
    def tool(self):
        """Create tool instance."""
        return WebSearchTool()

    def test_tool_name(self, tool):
        """Test tool name is correct."""
        assert tool.name == "web_search"

    def test_tool_description(self, tool):
        """Test tool has description."""
        assert tool.description
        assert "search" in tool.description.lower()
        assert "tavily" in tool.description.lower()

    def test_tool_integration_type(self, tool):
        """Test tool integration type."""
        assert tool.integration_type == "websearch"

    def test_no_oauth_required(self, tool):
        """Test that no OAuth scopes are required (uses API key)."""
        assert tool.required_scopes == []

    def test_parameters_schema(self, tool):
        """Test parameter schema structure."""
        schema = tool.get_parameters_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        # Required parameters
        assert "query" in schema["required"]
        assert len(schema["required"]) == 1  # Only query is required

        # Parameter definitions
        props = schema["properties"]
        assert "query" in props
        assert "max_results" in props
        assert "search_depth" in props
        assert "include_answer" in props
        assert "include_raw_content" in props

        # Type checks
        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"
        assert props["search_depth"]["type"] == "string"
        assert props["include_answer"]["type"] == "boolean"
        assert props["include_raw_content"]["type"] == "boolean"

        # Enum for search_depth
        assert props["search_depth"]["enum"] == ["basic", "advanced"]

        # Default values
        assert props["max_results"]["default"] == 5
        assert props["search_depth"]["default"] == "basic"
        assert props["include_answer"]["default"] is True
        assert props["include_raw_content"]["default"] is False

    def test_output_schema(self, tool):
        """Test output schema structure."""
        schema = tool.get_output_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "search_depth" in schema["properties"]
        assert "results" in schema["properties"]
        assert "result_count" in schema["properties"]
        assert "answer" in schema["properties"]

    def test_get_metadata(self, tool):
        """Test that get_metadata returns complete metadata."""
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
        """Create tool instance."""
        return WebSearchTool()

    @pytest.mark.asyncio
    async def test_error_when_api_key_not_configured(self, tool):
        """Test that HTTPException is raised when API key is not configured."""
        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = None

            with pytest.raises(HTTPException) as exc_info:
                await tool.execute(
                    access_token=None,
                    arguments={"query": "test query"},
                )

            assert exc_info.value.status_code == 503
            assert "Tavily API key not configured" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_successful_search_execution(self, tool):
        """Test successful search with mocked Tavily client."""
        mock_response = {
            "answer": "This is the AI-generated answer.",
            "results": [
                {
                    "title": "Test Result 1",
                    "url": "https://example.com/1",
                    "content": "This is the content snippet.",
                    "score": 0.95,
                },
                {
                    "title": "Test Result 2",
                    "url": "https://example.com/2",
                    "content": "Another content snippet.",
                    "score": 0.85,
                },
            ],
        }

        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            # Patch at tavily module level since it's imported inside the function
            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.return_value = mock_response
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                result = await tool.execute(
                    access_token=None,
                    arguments={"query": "test query"},
                )

                # Verify result structure
                assert result["query"] == "test query"
                assert result["search_depth"] == "basic"
                assert result["answer"] == "This is the AI-generated answer."
                assert result["result_count"] == 2
                assert len(result["results"]) == 2

                # Verify first result
                assert result["results"][0]["title"] == "Test Result 1"
                assert result["results"][0]["url"] == "https://example.com/1"
                assert result["results"][0]["content"] == "This is the content snippet."
                assert result["results"][0]["score"] == 0.95

                # Verify Tavily client was called correctly
                mock_client.search.assert_called_once_with(
                    query="test query",
                    max_results=5,
                    search_depth="basic",
                    include_answer=True,
                    include_raw_content=False,
                )

    @pytest.mark.asyncio
    async def test_max_results_clamping(self, tool):
        """Test that max_results is clamped to valid range."""
        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.return_value = {"results": []}
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                # Test clamping to minimum (1)
                await tool.execute(
                    access_token=None,
                    arguments={"query": "test", "max_results": 0},
                )
                _, kwargs = mock_client.search.call_args
                assert kwargs["max_results"] == 1

                # Test clamping to maximum (10)
                await tool.execute(
                    access_token=None,
                    arguments={"query": "test", "max_results": 20},
                )
                _, kwargs = mock_client.search.call_args
                assert kwargs["max_results"] == 10

    @pytest.mark.asyncio
    async def test_advanced_search_depth(self, tool):
        """Test that advanced search depth is passed correctly."""
        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.return_value = {"results": []}
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                await tool.execute(
                    access_token=None,
                    arguments={"query": "test", "search_depth": "advanced"},
                )

                _, kwargs = mock_client.search.call_args
                assert kwargs["search_depth"] == "advanced"

    @pytest.mark.asyncio
    async def test_include_raw_content(self, tool):
        """Test that raw_content is included when requested."""
        mock_response = {
            "results": [
                {
                    "title": "Test Result",
                    "url": "https://example.com/1",
                    "content": "Snippet",
                    "score": 0.9,
                    "raw_content": "Full page content here...",
                },
            ],
        }

        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.return_value = mock_response
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                result = await tool.execute(
                    access_token=None,
                    arguments={"query": "test", "include_raw_content": True},
                )

                assert result["results"][0]["raw_content"] == "Full page content here..."

    @pytest.mark.asyncio
    async def test_no_answer_when_disabled(self, tool):
        """Test that answer is not included when include_answer is False."""
        mock_response = {
            "answer": "Should not appear",
            "results": [],
        }

        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.return_value = mock_response
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                result = await tool.execute(
                    access_token=None,
                    arguments={"query": "test", "include_answer": False},
                )

                assert "answer" not in result

    @pytest.mark.asyncio
    async def test_tavily_api_error(self, tool):
        """Test that Tavily API errors are handled properly."""
        with patch("seer.tools.websearch.web_search.config") as mock_config:
            mock_config.tavily_api_key = "test-api-key"

            with patch.dict("sys.modules", {"tavily": MagicMock()}):
                import sys

                mock_client = MagicMock()
                mock_client.search.side_effect = Exception("API rate limit exceeded")
                sys.modules["tavily"].TavilyClient.return_value = mock_client

                with pytest.raises(HTTPException) as exc_info:
                    await tool.execute(
                        access_token=None,
                        arguments={"query": "test"},
                    )

                assert exc_info.value.status_code == 502
                assert "Web search failed" in str(exc_info.value.detail)
                assert "API rate limit exceeded" in str(exc_info.value.detail)


@pytest.mark.unit
class TestWebSearchToolRegistration:
    """Test that websearch tools can be registered."""

    def test_tools_can_be_imported(self):
        """Test that tools module can be imported."""
        from seer.tools.websearch import (
            WebSearchTool,
            register_websearch_tools,
        )

        assert WebSearchTool is not None
        assert callable(register_websearch_tools)

    def test_tool_registration(self):
        """Test that tool is registered in the registry."""
        from seer.tools.base import clear_registry, get_tool
        from seer.tools.websearch import register_websearch_tools

        # Clear and re-register to test
        clear_registry()
        register_websearch_tools()

        tool = get_tool("web_search")
        assert tool is not None
        assert tool.name == "web_search"
        assert isinstance(tool, WebSearchTool)
