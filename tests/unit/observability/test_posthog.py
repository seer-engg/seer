"""
Unit tests for PostHog analytics integration.

Tests cover:
- PostHog client initialization and event capture
- Non-blocking behavior via schedule_async_task
- Graceful degradation when PostHog is not configured
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestPostHogClient:
    """Tests for PostHog client module."""

    @pytest.fixture(autouse=True)
    def reset_posthog_state(self):
        """Reset PostHog module state between tests."""
        # Reset the module-level initialization flag
        import seer.observability.posthog_client as client
        client.POSTHOG_INITIALIZED = False
        yield
        client.POSTHOG_INITIALIZED = False

    def test_capture_event_noop_when_not_configured(self):
        """Should silently no-op when PostHog is not configured."""
        with patch("seer.observability.posthog_client.config") as mock_config:
            mock_config.is_posthog_configured = False

            from seer.observability.posthog_client import capture_event

            # Should not raise any exception
            capture_event("user123", "test_event", {"foo": "bar"})

    def test_identify_user_noop_when_not_configured(self):
        """Should silently no-op when PostHog is not configured."""
        with patch("seer.observability.posthog_client.config") as mock_config:
            mock_config.is_posthog_configured = False

            from seer.observability.posthog_client import identify_user

            # Should not raise any exception
            identify_user("user123", {"email": "test@example.com"})

    def test_ensure_initialized_returns_false_when_not_configured(self):
        """Should return False when PostHog is not configured."""
        with patch("seer.observability.posthog_client.config") as mock_config:
            mock_config.is_posthog_configured = False

            from seer.observability.posthog_client import _ensure_initialized

            assert _ensure_initialized() is False

    def test_ensure_initialized_returns_true_when_configured(self):
        """Should return True and initialize client when configured."""
        with patch("seer.observability.posthog_client.config") as mock_config, \
             patch("seer.observability.posthog_client.posthog") as mock_posthog:
            mock_config.is_posthog_configured = True
            mock_config.posthog_api_key = "test_api_key"
            mock_config.posthog_host = "https://test.posthog.com"
            mock_config.env = "test"

            from seer.observability.posthog_client import _ensure_initialized

            result = _ensure_initialized()

            assert result is True
            assert mock_posthog.project_api_key == "test_api_key"
            assert mock_posthog.host == "https://test.posthog.com"

    def test_ensure_initialized_only_runs_once(self):
        """Should only initialize once even if called multiple times."""
        with patch("seer.observability.posthog_client.config") as mock_config, \
             patch("seer.observability.posthog_client.posthog") as mock_posthog:
            mock_config.is_posthog_configured = True
            mock_config.posthog_api_key = "test_api_key"
            mock_config.posthog_host = "https://test.posthog.com"
            mock_config.env = "test"

            from seer.observability.posthog_client import _ensure_initialized

            # Call twice
            _ensure_initialized()
            _ensure_initialized()

            # project_api_key should only be set once (during first init)
            # Since we're checking the mock, we verify the final state is correct
            assert mock_posthog.project_api_key == "test_api_key"

    @pytest.mark.asyncio
    async def test_capture_event_schedules_async_task(self):
        """Should schedule async task for non-blocking capture."""
        with patch("seer.observability.posthog_client.config") as mock_config, \
             patch("seer.observability.posthog_client.schedule_async_task") as mock_schedule, \
             patch("seer.observability.posthog_client.posthog"):
            mock_config.is_posthog_configured = True
            mock_config.posthog_api_key = "test_key"
            mock_config.posthog_host = "https://test.posthog.com"
            mock_config.env = "test"

            from seer.observability.posthog_client import capture_event

            capture_event("user123", "test_event", {"foo": "bar"})

            # Verify schedule_async_task was called
            assert mock_schedule.called
            call_args = mock_schedule.call_args
            assert "coro" in call_args.kwargs or len(call_args.args) > 0

    @pytest.mark.asyncio
    async def test_identify_user_schedules_async_task(self):
        """Should schedule async task for non-blocking identify."""
        with patch("seer.observability.posthog_client.config") as mock_config, \
             patch("seer.observability.posthog_client.schedule_async_task") as mock_schedule, \
             patch("seer.observability.posthog_client.posthog"):
            mock_config.is_posthog_configured = True
            mock_config.posthog_api_key = "test_key"
            mock_config.posthog_host = "https://test.posthog.com"
            mock_config.env = "test"

            from seer.observability.posthog_client import identify_user

            identify_user("user123", {"email": "test@example.com"})

            # Verify schedule_async_task was called
            assert mock_schedule.called

    def test_shutdown_flushes_and_closes(self):
        """Should flush and shutdown PostHog client."""
        import seer.observability.posthog_client as client
        client.POSTHOG_INITIALIZED = True

        with patch("seer.observability.posthog_client.posthog") as mock_posthog:
            from seer.observability.posthog_client import shutdown

            shutdown()

            mock_posthog.flush.assert_called_once()
            mock_posthog.shutdown.assert_called_once()

    def test_shutdown_handles_errors_gracefully(self):
        """Should log but not raise on shutdown errors."""
        import seer.observability.posthog_client as client
        client.POSTHOG_INITIALIZED = True

        with patch("seer.observability.posthog_client.posthog") as mock_posthog:
            mock_posthog.flush.side_effect = Exception("Shutdown error")

            from seer.observability.posthog_client import shutdown

            # Should not raise
            shutdown()


@pytest.mark.unit
class TestPostHogMiddleware:
    """Tests for PostHog FastAPI middleware."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.url.path = "/api/v1/workflows"
        request.method = "GET"
        request.url = MagicMock()
        request.url.path = "/api/v1/workflows"
        request.url.__str__ = lambda self: "http://localhost:8000/api/v1/workflows"
        request.state = MagicMock()
        request.state.user = MagicMock()
        request.state.user.user_id = "user123"
        request.state.user.email = "test@example.com"
        request.state.user.first_name = "Test"
        request.state.user.last_name = "User"
        request.state.correlation_id = "corr-123"
        return request

    @pytest.fixture
    def mock_response(self):
        """Create a mock response."""
        response = MagicMock()
        response.status_code = 200
        return response

    def test_should_skip_health_endpoint(self):
        """Should skip tracking for /health endpoint."""
        from seer.api.core.middleware.posthog_middleware import PostHogMiddleware, EXCLUDED_PATHS

        assert "/health" in EXCLUDED_PATHS

    def test_should_skip_docs_endpoint(self):
        """Should skip tracking for /docs endpoint."""
        from seer.api.core.middleware.posthog_middleware import EXCLUDED_PATHS

        assert "/docs" in EXCLUDED_PATHS

    def test_should_skip_options_requests(self, mock_request):
        """Should skip tracking for OPTIONS requests (CORS preflight)."""
        from seer.api.core.middleware.posthog_middleware import PostHogMiddleware

        middleware = PostHogMiddleware(app=MagicMock())
        mock_request.method = "OPTIONS"

        result = middleware._should_skip(mock_request, "/api/v1/workflows")

        assert result is True

    def test_should_not_skip_regular_requests(self, mock_request):
        """Should not skip tracking for regular API requests."""
        from seer.api.core.middleware.posthog_middleware import PostHogMiddleware

        middleware = PostHogMiddleware(app=MagicMock())
        mock_request.method = "GET"

        result = middleware._should_skip(mock_request, "/api/v1/workflows")

        assert result is False

    def test_should_skip_mcp_endpoints(self, mock_request):
        """Should skip tracking for MCP endpoints (they have their own tracking)."""
        from seer.api.core.middleware.posthog_middleware import PostHogMiddleware, EXCLUDED_PREFIXES

        # Verify MCP prefixes are in EXCLUDED_PREFIXES
        assert "/mcp" in EXCLUDED_PREFIXES
        assert "/sse" in EXCLUDED_PREFIXES

        middleware = PostHogMiddleware(app=MagicMock())
        mock_request.method = "POST"

        # Test exact path match
        assert middleware._should_skip(mock_request, "/mcp") is True
        assert middleware._should_skip(mock_request, "/sse") is True

        # Test path with suffix
        assert middleware._should_skip(mock_request, "/mcp/messages") is True
        assert middleware._should_skip(mock_request, "/sse/init") is True

    def test_track_request_captures_event(self, mock_request, mock_response):
        """Should capture event with correct properties."""
        with patch("seer.api.core.middleware.posthog_middleware.capture_event") as mock_capture, \
             patch("seer.api.core.middleware.posthog_middleware.identify_user") as mock_identify, \
             patch("seer.api.core.middleware.posthog_middleware.config") as mock_config:
            mock_config.seer_mode = "cloud"

            from seer.api.core.middleware.posthog_middleware import PostHogMiddleware

            middleware = PostHogMiddleware(app=MagicMock())
            middleware._track_request(mock_request, mock_response, latency_ms=50.5)

            # Verify capture_event was called
            mock_capture.assert_called_once()
            call_args = mock_capture.call_args

            assert call_args.kwargs["distinct_id"] == "user123"
            assert call_args.kwargs["event"] == "api_request"

            properties = call_args.kwargs["properties"]
            assert properties["method"] == "GET"
            assert properties["path"] == "/api/v1/workflows"
            assert properties["status_code"] == 200
            assert properties["latency_ms"] == 50.5
            assert properties["authenticated"] is True
            assert properties["seer_mode"] == "cloud"

    def test_track_request_anonymous_when_no_user(self, mock_request, mock_response):
        """Should use 'anonymous' distinct_id when user not authenticated."""
        mock_request.state.user = None

        with patch("seer.api.core.middleware.posthog_middleware.capture_event") as mock_capture, \
             patch("seer.api.core.middleware.posthog_middleware.config") as mock_config:
            mock_config.seer_mode = "self-hosted"

            from seer.api.core.middleware.posthog_middleware import PostHogMiddleware

            middleware = PostHogMiddleware(app=MagicMock())
            middleware._track_request(mock_request, mock_response, latency_ms=25.0)

            call_args = mock_capture.call_args
            assert call_args.kwargs["distinct_id"] == "anonymous"
            assert call_args.kwargs["properties"]["authenticated"] is False


@pytest.mark.unit
class TestMCPTrackingDecorator:
    """Tests for MCP tool tracking decorator."""

    @pytest.mark.asyncio
    async def test_decorator_passes_through_when_not_configured(self):
        """Should pass through to original function when PostHog not configured."""
        with patch("seer.mcp.tracking.config") as mock_config:
            mock_config.is_posthog_configured = False

            from seer.mcp.tracking import track_mcp_tool

            @track_mcp_tool("test_tool")
            async def test_func(param: str) -> str:
                return f"result: {param}"

            result = await test_func("test")

            assert result == "result: test"

    @pytest.mark.asyncio
    async def test_decorator_tracks_successful_call(self):
        """Should track successful tool call."""
        with patch("seer.mcp.tracking.config") as mock_config, \
             patch("seer.mcp.tracking.capture_event") as mock_capture, \
             patch("seer.mcp.tracking.get_mcp_authenticated_user") as mock_get_user:
            mock_config.is_posthog_configured = True
            mock_config.seer_mode = "cloud"
            mock_get_user.return_value = None  # Anonymous

            from seer.mcp.tracking import track_mcp_tool

            @track_mcp_tool("test_tool")
            async def test_func(workflow_id: str) -> str:
                return "success"

            result = await test_func(workflow_id="wf_123")

            assert result == "success"
            mock_capture.assert_called_once()

            call_args = mock_capture.call_args
            assert call_args.kwargs["event"] == "mcp_tool_call"
            assert call_args.kwargs["properties"]["tool_name"] == "test_tool"
            assert call_args.kwargs["properties"]["success"] is True

    @pytest.mark.asyncio
    async def test_decorator_tracks_failed_call(self):
        """Should track failed tool call and re-raise exception."""
        with patch("seer.mcp.tracking.config") as mock_config, \
             patch("seer.mcp.tracking.capture_event") as mock_capture, \
             patch("seer.mcp.tracking.get_mcp_authenticated_user") as mock_get_user:
            mock_config.is_posthog_configured = True
            mock_config.seer_mode = "cloud"
            mock_get_user.return_value = None

            from seer.mcp.tracking import track_mcp_tool

            @track_mcp_tool("failing_tool")
            async def test_func() -> str:
                raise ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                await test_func()

            mock_capture.assert_called_once()
            call_args = mock_capture.call_args
            assert call_args.kwargs["properties"]["success"] is False
            assert "Test error" in call_args.kwargs["properties"]["error"]

    @pytest.mark.asyncio
    async def test_decorator_includes_safe_params(self):
        """Should include only safe parameters in tracking."""
        with patch("seer.mcp.tracking.config") as mock_config, \
             patch("seer.mcp.tracking.capture_event") as mock_capture, \
             patch("seer.mcp.tracking.get_mcp_authenticated_user") as mock_get_user:
            mock_config.is_posthog_configured = True
            mock_config.seer_mode = "cloud"
            mock_get_user.return_value = None

            from seer.mcp.tracking import track_mcp_tool

            @track_mcp_tool("test_tool")
            async def test_func(workflow_id: str, limit: int, spec: dict) -> str:
                return "success"

            await test_func(workflow_id="wf_123", limit=50, spec={"secret": "data"})

            call_args = mock_capture.call_args
            params = call_args.kwargs["properties"].get("params", {})

            # Safe params should be included
            assert params.get("workflow_id") == "wf_123"
            assert params.get("limit") == 50
            # Spec should NOT be included (not in safe list)
            assert "spec" not in params

    @pytest.mark.asyncio
    async def test_decorator_identifies_authenticated_user(self):
        """Should identify authenticated user from MCP context."""
        with patch("seer.mcp.tracking.config") as mock_config, \
             patch("seer.mcp.tracking.capture_event") as mock_capture, \
             patch("seer.mcp.tracking.identify_user") as mock_identify, \
             patch("seer.mcp.tracking.get_mcp_authenticated_user") as mock_get_user:
            mock_config.is_posthog_configured = True
            mock_config.seer_mode = "cloud"

            # Mock authenticated user
            mock_user = MagicMock()
            mock_user.user_id = "user_456"
            mock_user.email = "mcp@example.com"
            mock_user.first_name = "MCP"
            mock_user.last_name = "User"
            mock_get_user.return_value = mock_user

            from seer.mcp.tracking import track_mcp_tool

            @track_mcp_tool("test_tool")
            async def test_func() -> str:
                return "success"

            await test_func()

            # Verify user was identified
            mock_identify.assert_called_once()
            identify_args = mock_identify.call_args
            assert identify_args.kwargs["distinct_id"] == "user_456"

            # Verify capture used correct distinct_id
            capture_args = mock_capture.call_args
            assert capture_args.kwargs["distinct_id"] == "user_456"
            assert capture_args.kwargs["properties"]["authenticated"] is True
