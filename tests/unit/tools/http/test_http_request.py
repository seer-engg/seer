"""Unit tests for HTTP request tool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from seer.tools.http.request import HttpRequestTool


@pytest.mark.unit
class TestHttpRequestToolMetadata:
    @pytest.fixture
    def tool(self):
        return HttpRequestTool()

    def test_tool_name(self, tool):
        assert tool.name == "http_request"

    def test_parameters_schema(self, tool):
        schema = tool.get_parameters_schema()
        assert "url" in schema["required"]
        props = schema["properties"]
        assert "method" in props
        assert "url" in props
        assert "headers" in props
        assert "body" in props

    def test_output_schema(self, tool):
        schema = tool.get_output_schema()
        props = schema["properties"]
        assert "status_code" in props
        assert "body" in props
        assert "headers" in props


@pytest.mark.unit
class TestHttpRequestToolExecute:
    @pytest.fixture
    def tool(self):
        return HttpRequestTool()

    @pytest.fixture
    def mock_response(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {"key": "value"}
        resp.text = '{"key": "value"}'
        return resp

    async def test_get_request(self, tool, mock_response):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("seer.tools.http.request.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(None, {"url": "https://example.com/api", "method": "GET"})

        assert result["status_code"] == 200
        assert result["body"] == {"key": "value"}
        mock_client.request.assert_called_once_with(
            "GET", "https://example.com/api", headers={}, params={}
        )

    async def test_post_with_body(self, tool, mock_response):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("seer.tools.http.request.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(None, {
                "url": "https://example.com/api",
                "method": "POST",
                "body": {"data": "test"},
            })

        assert result["status_code"] == 200
        mock_client.request.assert_called_once_with(
            "POST", "https://example.com/api", headers={}, params={}, json={"data": "test"}
        )

    async def test_non_json_response(self, tool):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-type": "text/plain"}
        resp.json.side_effect = ValueError("not json")
        resp.text = "plain text response"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("seer.tools.http.request.httpx.AsyncClient", return_value=mock_client):
            result = await tool.execute(None, {"url": "https://example.com"})

        assert result["body"] == "plain text response"

    async def test_default_method_is_get(self, tool, mock_response):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("seer.tools.http.request.httpx.AsyncClient", return_value=mock_client):
            await tool.execute(None, {"url": "https://example.com"})

        mock_client.request.assert_called_once_with(
            "GET", "https://example.com", headers={}, params={}
        )

    async def test_custom_headers(self, tool, mock_response):
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("seer.tools.http.request.httpx.AsyncClient", return_value=mock_client):
            await tool.execute(None, {
                "url": "https://example.com",
                "headers": {"Authorization": "Bearer token123"},
            })

        mock_client.request.assert_called_once_with(
            "GET", "https://example.com", headers={"Authorization": "Bearer token123"}, params={}
        )
