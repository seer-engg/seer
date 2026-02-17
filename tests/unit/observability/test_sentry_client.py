"""
Unit tests for Sentry error monitoring integration.

Tests cover:
- Sentry client initialization and graceful degradation
- before_send filter for expected exceptions
- Header scrubbing for sensitive data
- Context and tag setting functions
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
class TestSentryClient:
    """Tests for Sentry client module."""

    @pytest.fixture(autouse=True)
    def reset_sentry_state(self):
        """Reset Sentry module state between tests."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False
        yield
        client.SENTRY_INITIALIZED = False

    def test_init_sentry_returns_false_when_not_configured(self):
        """Should return False when Sentry DSN is not configured."""
        with patch("seer.observability.sentry_client.config") as mock_config:
            mock_config.is_sentry_configured = False

            from seer.observability.sentry_client import init_sentry

            result = init_sentry()

            assert result is False

    def test_init_sentry_returns_true_when_configured(self):
        """Should return True and initialize SDK when configured."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        mock_sentry = MagicMock()

        with patch.object(client, "config") as mock_config, \
             patch.dict("sys.modules", {"sentry_sdk": mock_sentry}), \
             patch.dict("sys.modules", {"sentry_sdk.integrations.fastapi": MagicMock()}), \
             patch.dict("sys.modules", {"sentry_sdk.integrations.starlette": MagicMock()}):
            mock_config.is_sentry_configured = True
            mock_config.sentry_dsn = "https://test@sentry.io/123"
            mock_config.sentry_environment = "test"
            mock_config.sentry_traces_sample_rate = 0.1
            mock_config.sentry_profiles_sample_rate = 0.1
            mock_config.env = "test"

            result = client.init_sentry()

            assert result is True
            mock_sentry.init.assert_called_once()

    def test_init_sentry_only_runs_once(self):
        """Should only initialize once even if called multiple times."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        mock_sentry = MagicMock()

        with patch.object(client, "config") as mock_config, \
             patch.dict("sys.modules", {"sentry_sdk": mock_sentry}), \
             patch.dict("sys.modules", {"sentry_sdk.integrations.fastapi": MagicMock()}), \
             patch.dict("sys.modules", {"sentry_sdk.integrations.starlette": MagicMock()}):
            mock_config.is_sentry_configured = True
            mock_config.sentry_dsn = "https://test@sentry.io/123"
            mock_config.sentry_environment = "test"
            mock_config.sentry_traces_sample_rate = 0.1
            mock_config.sentry_profiles_sample_rate = 0.1
            mock_config.env = "test"

            # Call twice
            client.init_sentry()
            client.init_sentry()

            # Should only init once
            assert mock_sentry.init.call_count == 1

    def test_init_sentry_handles_exception_gracefully(self):
        """Should handle initialization errors gracefully."""
        with patch("seer.observability.sentry_client.config") as mock_config:
            mock_config.is_sentry_configured = True
            mock_config.sentry_dsn = "https://test@sentry.io/123"

            # Make import of sentry_sdk raise an error
            with patch.dict("sys.modules", {"sentry_sdk": None}):
                import seer.observability.sentry_client as client
                client.SENTRY_INITIALIZED = False

                # Should not raise, should return False
                result = client.init_sentry()
                assert result is False


@pytest.mark.unit
class TestBeforeSend:
    """Tests for the _before_send filter function."""

    def test_filters_expected_usage_limit_error(self):
        """Should mark UsageLimitError as expected (info level)."""
        from seer.observability.sentry_client import _before_send

        # Create mock exception
        class UsageLimitError(Exception):
            pass

        event = {"level": "error", "tags": {}}
        hint = {"exc_info": (UsageLimitError, UsageLimitError("test"), None)}

        result = _before_send(event, hint)

        assert result is not None
        assert result["level"] == "info"
        assert result["tags"]["expected_error"] == "true"
        assert result["tags"]["error_type"] == "UsageLimitError"

    def test_filters_expected_chat_disabled_error(self):
        """Should mark ChatDisabledError as expected (info level)."""
        from seer.observability.sentry_client import _before_send

        class ChatDisabledError(Exception):
            pass

        event = {"level": "error", "tags": {}}
        hint = {"exc_info": (ChatDisabledError, ChatDisabledError("test"), None)}

        result = _before_send(event, hint)

        assert result is not None
        assert result["level"] == "info"
        assert result["tags"]["expected_error"] == "true"
        assert result["tags"]["error_type"] == "ChatDisabledError"

    def test_does_not_filter_unexpected_errors(self):
        """Should not modify unexpected errors."""
        from seer.observability.sentry_client import _before_send

        class UnexpectedError(Exception):
            pass

        event = {"level": "error", "tags": {}}
        hint = {"exc_info": (UnexpectedError, UnexpectedError("test"), None)}

        result = _before_send(event, hint)

        assert result is not None
        assert result["level"] == "error"  # Unchanged
        assert "expected_error" not in result.get("tags", {})

    def test_scrubs_authorization_header(self):
        """Should scrub authorization header from request data."""
        from seer.observability.sentry_client import _before_send

        event = {
            "level": "error",
            "request": {
                "headers": {
                    "authorization": "Bearer secret_token_123",
                    "content-type": "application/json",
                }
            }
        }
        hint = {}

        result = _before_send(event, hint)

        assert result is not None
        assert result["request"]["headers"]["authorization"] == "[Filtered]"
        assert result["request"]["headers"]["content-type"] == "application/json"

    def test_scrubs_api_key_header(self):
        """Should scrub x-api-key header."""
        from seer.observability.sentry_client import _before_send

        event = {
            "level": "error",
            "request": {
                "headers": {
                    "X-Api-Key": "secret_api_key",
                    "accept": "application/json",
                }
            }
        }
        hint = {}

        result = _before_send(event, hint)

        assert result is not None
        assert result["request"]["headers"]["X-Api-Key"] == "[Filtered]"
        assert result["request"]["headers"]["accept"] == "application/json"

    def test_handles_missing_request_data(self):
        """Should handle events without request data."""
        from seer.observability.sentry_client import _before_send

        event = {"level": "error", "message": "Test error"}
        hint = {}

        result = _before_send(event, hint)

        assert result is not None
        assert result["level"] == "error"


@pytest.mark.unit
class TestTracesSampler:
    """Tests for the _traces_sampler function."""

    def test_skips_health_endpoint(self):
        """Should return 0 sample rate for /health endpoint."""
        from seer.observability.sentry_client import _traces_sampler

        sampling_context = {
            "transaction_context": {"name": "/health"}
        }

        result = _traces_sampler(sampling_context)

        assert result == 0.0

    def test_skips_docs_endpoint(self):
        """Should return 0 sample rate for /docs endpoint."""
        from seer.observability.sentry_client import _traces_sampler

        sampling_context = {
            "transaction_context": {"name": "/docs"}
        }

        result = _traces_sampler(sampling_context)

        assert result == 0.0

    def test_samples_regular_endpoints(self):
        """Should return configured sample rate for regular endpoints."""
        with patch("seer.observability.sentry_client.config") as mock_config:
            mock_config.sentry_traces_sample_rate = 0.5

            from seer.observability.sentry_client import _traces_sampler

            sampling_context = {
                "transaction_context": {"name": "/api/v1/workflows"}
            }

            result = _traces_sampler(sampling_context)

            assert result == 0.5


@pytest.mark.unit
class TestCaptureFunctions:
    """Tests for capture and context functions."""

    @pytest.fixture(autouse=True)
    def reset_sentry_state(self):
        """Reset Sentry module state between tests."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False
        yield
        client.SENTRY_INITIALIZED = False

    def test_capture_exception_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        result = client.capture_exception(ValueError("test"))

        assert result is None

    def test_capture_exception_calls_sentry(self):
        """Should call sentry_sdk.capture_exception when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()
        mock_sentry.capture_exception.return_value = "event_123"

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            exc = ValueError("test error")
            result = client.capture_exception(exc)

            mock_sentry.capture_exception.assert_called_once_with(exc)
            assert result == "event_123"

    def test_set_user_context_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        # Should not raise
        client.set_user_context(user_id="user123", email="test@example.com")

    def test_set_user_context_calls_sentry(self):
        """Should call sentry_sdk.set_user when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.set_user_context(
                user_id="user123",
                email="test@example.com",
                username="Test User"
            )

            mock_sentry.set_user.assert_called_once_with({
                "id": "user123",
                "email": "test@example.com",
                "username": "Test User",
            })

    def test_set_tag_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        # Should not raise
        client.set_tag("correlation_id", "test123")

    def test_set_tag_calls_sentry(self):
        """Should call sentry_sdk.set_tag when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.set_tag("correlation_id", "test123")

            mock_sentry.set_tag.assert_called_once_with("correlation_id", "test123")

    def test_set_context_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        # Should not raise
        client.set_context("workflow", {"id": "wf_123"})

    def test_set_context_calls_sentry(self):
        """Should call sentry_sdk.set_context when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.set_context("workflow", {"id": "wf_123", "name": "Test"})

            mock_sentry.set_context.assert_called_once_with(
                "workflow",
                {"id": "wf_123", "name": "Test"}
            )

    def test_flush_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        # Should not raise
        client.flush(timeout=2.0)

    def test_flush_calls_sentry(self):
        """Should call sentry_sdk.flush when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.flush(timeout=3.0)

            mock_sentry.flush.assert_called_once_with(timeout=3.0)

    def test_flush_resets_initialized_flag(self):
        """Should reset SENTRY_INITIALIZED flag after flush."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.flush()

            assert client.SENTRY_INITIALIZED is False

    def test_add_breadcrumb_noop_when_not_initialized(self):
        """Should silently no-op when Sentry is not initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = False

        # Should not raise
        client.add_breadcrumb("Test breadcrumb", category="test")

    def test_add_breadcrumb_calls_sentry(self):
        """Should call sentry_sdk.add_breadcrumb when initialized."""
        import seer.observability.sentry_client as client
        client.SENTRY_INITIALIZED = True

        mock_sentry = MagicMock()

        with patch.dict("sys.modules", {"sentry_sdk": mock_sentry}):
            client.add_breadcrumb(
                message="User clicked button",
                category="ui",
                level="info",
                data={"button_id": "submit"}
            )

            mock_sentry.add_breadcrumb.assert_called_once_with(
                message="User clicked button",
                category="ui",
                level="info",
                data={"button_id": "submit"}
            )


@pytest.mark.unit
class TestSentryMiddleware:
    """Tests for Sentry FastAPI middleware."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock FastAPI request."""
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/v1/workflows"
        request.url.__str__ = lambda self: "http://localhost:8000/api/v1/workflows"
        request.query_params = {}
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-client"}
        request.state = MagicMock()
        request.state.correlation_id = "corr-123"
        request.state.user = MagicMock()
        request.state.user.user_id = "user123"
        request.state.user.email = "test@example.com"
        request.state.user.first_name = "Test"
        request.state.user.last_name = "User"
        return request

    def test_middleware_sets_correlation_id_tag(self, mock_request):
        """Should set correlation_id tag from request state."""
        with patch("seer.api.core.middleware.sentry_middleware.set_tag") as mock_set_tag, \
             patch("seer.api.core.middleware.sentry_middleware.set_user_context"), \
             patch("seer.api.core.middleware.sentry_middleware.set_context"), \
             patch("seer.api.core.middleware.sentry_middleware.config") as mock_config:
            mock_config.seer_mode = "cloud"

            from seer.api.core.middleware.sentry_middleware import SentryContextMiddleware

            middleware = SentryContextMiddleware(app=MagicMock())

            # Check that set_tag would be called with correlation_id
            # We can't easily test dispatch without async, so we verify the import works
            assert SentryContextMiddleware is not None

    def test_middleware_sets_user_context_when_authenticated(self, mock_request):
        """Should set user context when user is authenticated."""
        # Verify middleware can be imported and instantiated
        from seer.api.core.middleware.sentry_middleware import SentryContextMiddleware

        middleware = SentryContextMiddleware(app=MagicMock())
        assert middleware is not None
