"""Unit tests for Google API base client retry logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from seer.tools.google.base import GoogleAPIClient, TRANSIENT_ERRORS


class MockGoogleTool(GoogleAPIClient):
    """Mock Google tool for testing."""

    name = "test_google_tool"
    description = "Test Google tool"
    required_scopes = ["https://www.googleapis.com/auth/test"]
    integration_type = "google"

    async def execute(self, credentials, arguments, context=None):
        """Mock execute method."""
        return {"result": "success"}


class TestTransientErrors:
    """Tests for transient error configuration."""

    def test_transient_errors_includes_remote_protocol_error(self):
        """Test that RemoteProtocolError is in TRANSIENT_ERRORS."""
        assert httpx.RemoteProtocolError in TRANSIENT_ERRORS

    def test_transient_errors_includes_connect_error(self):
        """Test that ConnectError is in TRANSIENT_ERRORS."""
        assert httpx.ConnectError in TRANSIENT_ERRORS

    def test_transient_errors_includes_connect_timeout(self):
        """Test that ConnectTimeout is in TRANSIENT_ERRORS."""
        assert httpx.ConnectTimeout in TRANSIENT_ERRORS


@pytest.mark.asyncio
class TestMakeRequestRetry:
    """Tests for _make_request retry behavior."""

    async def test_make_request_success_no_retry(self):
        """Test successful request without retry."""
        tool = MockGoogleTool()

        mock_response = MagicMock()
        mock_response.is_error = False

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            result = await tool._make_request(
                "GET",
                "https://api.google.com/test",
                "test_token"
            )

            assert result == mock_response
            mock_execute.assert_called_once()

    async def test_make_request_timeout_raises_504(self):
        """Test that timeout raises 504 HTTPException."""
        tool = MockGoogleTool()

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = httpx.TimeoutException("Request timed out")

            with pytest.raises(HTTPException) as exc_info:
                await tool._make_request(
                    "GET",
                    "https://api.google.com/test",
                    "test_token",
                    timeout=5.0
                )

            assert exc_info.value.status_code == 504
            assert "timed out" in exc_info.value.detail.lower()

    async def test_make_request_transient_error_after_retries_raises_503(self):
        """Test that transient error after retries raises 503 HTTPException."""
        tool = MockGoogleTool()

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = httpx.RemoteProtocolError("Server closed connection")

            with pytest.raises(HTTPException) as exc_info:
                await tool._make_request(
                    "GET",
                    "https://api.google.com/test",
                    "test_token"
                )

            assert exc_info.value.status_code == 503
            assert "network error after retries" in exc_info.value.detail.lower()

    async def test_make_request_connect_error_after_retries_raises_503(self):
        """Test that ConnectError after retries raises 503 HTTPException."""
        tool = MockGoogleTool()

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = httpx.ConnectError("Failed to connect")

            with pytest.raises(HTTPException) as exc_info:
                await tool._make_request(
                    "GET",
                    "https://api.google.com/test",
                    "test_token"
                )

            assert exc_info.value.status_code == 503

    async def test_make_request_unexpected_error_raises_500(self):
        """Test that unexpected error raises 500 HTTPException."""
        tool = MockGoogleTool()

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(HTTPException) as exc_info:
                await tool._make_request(
                    "GET",
                    "https://api.google.com/test",
                    "test_token"
                )

            assert exc_info.value.status_code == 500

    async def test_make_request_http_exception_propagated(self):
        """Test that HTTPException is propagated without wrapping."""
        tool = MockGoogleTool()

        with patch.object(tool, '_execute_request_with_retry', new_callable=AsyncMock) as mock_execute:
            mock_execute.side_effect = HTTPException(status_code=401, detail="Unauthorized")

            with pytest.raises(HTTPException) as exc_info:
                await tool._make_request(
                    "GET",
                    "https://api.google.com/test",
                    "test_token"
                )

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Unauthorized"


@pytest.mark.asyncio
class TestExecuteRequestWithRetry:
    """Tests for _execute_request_with_retry method."""

    async def test_retry_on_remote_protocol_error(self):
        """Test that RemoteProtocolError triggers retry."""
        tool = MockGoogleTool()

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.RemoteProtocolError("Server closed connection")
            mock_resp = MagicMock()
            mock_resp.is_error = False
            return mock_resp

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = mock_request
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await tool._execute_request_with_retry(
                method="GET",
                url="https://api.google.com/test",
                headers={"Authorization": "Bearer token"},
                params=None,
                json_body=None,
                content=None,
                timeout_value=30.0,
            )

            assert result is not None
            assert call_count == 3  # 2 failures + 1 success

    async def test_retry_exhausted_raises_error(self):
        """Test that error is raised when all retries are exhausted."""
        tool = MockGoogleTool()

        async def always_fail(*args, **kwargs):
            raise httpx.RemoteProtocolError("Server closed connection")

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = always_fail
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(httpx.RemoteProtocolError):
                await tool._execute_request_with_retry(
                    method="GET",
                    url="https://api.google.com/test",
                    headers={"Authorization": "Bearer token"},
                    params=None,
                    json_body=None,
                    content=None,
                    timeout_value=30.0,
                )

    async def test_api_error_not_retried(self):
        """Test that API errors (4xx, 5xx) are not retried."""
        tool = MockGoogleTool()

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.is_error = True
            mock_resp.status_code = 404
            mock_resp.text = "Not Found"
            return mock_resp

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.request = mock_request
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await tool._execute_request_with_retry(
                    method="GET",
                    url="https://api.google.com/test",
                    headers={"Authorization": "Bearer token"},
                    params=None,
                    json_body=None,
                    content=None,
                    timeout_value=30.0,
                )

            # Should only be called once - no retry for API errors
            assert call_count == 1
            assert exc_info.value.status_code == 404
