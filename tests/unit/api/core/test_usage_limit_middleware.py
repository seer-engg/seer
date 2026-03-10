"""
Unit tests for usage limit middleware.

Tests:
- UsageLimitMiddleware.dispatch: Limit enforcement
- Workflow creation limit check
- Monthly run limit check
- Polling interval validation
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.core.middleware.usage_limit import UsageLimitMiddleware


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock()
    request.method = "POST"
    request.url.path = "/api/v1/workflows"
    request.state = MagicMock()
    request.state.organization = None  # Organization context (may be None for personal workspace)
    return request

# Note: mock_user fixture is provided by tests/unit/conftest.py


@pytest.fixture
def mock_limits():
    """Create mock tier limits."""
    limits = MagicMock()
    limits.workflows = 10
    limits.runs_monthly = 100
    limits.poll_min_interval_seconds = 60
    limits.has_unlimited_workflows = False
    limits.has_unlimited_runs = False
    return limits


@pytest.fixture
def unlimited_limits():
    """Create mock unlimited tier limits."""
    limits = MagicMock()
    limits.workflows = 999999
    limits.runs_monthly = 999999
    limits.poll_min_interval_seconds = 1
    limits.has_unlimited_workflows = True
    limits.has_unlimited_runs = True
    return limits


# =============================================================================
# Dispatch Basic Tests
# =============================================================================


@pytest.mark.unit
class TestDispatchBasic:
    """Tests for basic dispatch functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_skips_options_request(self, mock_request):
        """Test that OPTIONS requests are passed through."""
        mock_request.method = "OPTIONS"
        call_next = AsyncMock(return_value=MagicMock())

        middleware = UsageLimitMiddleware(app=MagicMock())

        await middleware.dispatch(mock_request, call_next)

        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_skips_public_path(self, mock_request):
        """Test that public paths are passed through."""
        mock_request.method = "GET"
        mock_request.url.path = "/health"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public:
            mock_is_public.return_value = True

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_requires_authentication(self, mock_request):
        """Test that unauthenticated requests return 401."""
        mock_request.state.db_user = None
        mock_request.url.path = "/api/v1/workflows"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public:
            mock_is_public.return_value = False

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_dispatch_payment_exempt_path_skips_limits(self, mock_request, mock_user):
        """Test that payment-exempt paths skip usage limit checks."""
        mock_request.state.db_user = mock_user
        mock_request.method = "GET"
        mock_request.url.path = "/api/subscriptions/checkout"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = True

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should pass through without checking limits
            call_next.assert_called_once()
            # get_effective_limits should not be called
            mock_get_limits.assert_not_called()

    @pytest.mark.asyncio
    async def test_payment_exempt_path_requires_auth(self, mock_request):
        """Test that payment-exempt paths still require authentication."""
        mock_request.state.db_user = None
        mock_request.method = "GET"
        mock_request.url.path = "/api/usage"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = True

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            # Should return 401 (not allowed without auth)
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_usage_analytics_path_is_payment_exempt(self, mock_request, mock_user):
        """Test that /api/usage/analytics/* paths are payment-exempt."""
        mock_request.state.db_user = mock_user
        mock_request.method = "GET"
        mock_request.url.path = "/api/usage/analytics/daily"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = True

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should pass through
            call_next.assert_called_once()
            # Limits should not be checked
            mock_get_limits.assert_not_called()


    @pytest.mark.asyncio
    async def test_non_payment_exempt_path_enforces_limits(self, mock_request, mock_user, mock_limits):
        """Test that regular paths still enforce usage limits."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_workflow_count") as mock_get_count, \
             patch("seer.api.core.middleware.usage_limit.resolve_user_tier") as mock_resolve_tier:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 10  # At limit
            mock_resolve_tier.return_value = MagicMock(value="free")

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            # Should be blocked with 402
            assert result.status_code == 402
            # Limits should have been checked
            mock_get_limits.assert_called_once()


# =============================================================================
# Workflow Creation Limit Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowCreationLimit:
    """Tests for workflow creation limit enforcement."""

    @pytest.mark.asyncio
    async def test_workflow_creation_within_limit(self, mock_request, mock_user, mock_limits):
        """Test that workflow creation is allowed within limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_workflow_count") as mock_get_count:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 5  # Under limit of 10

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_creation_at_limit(self, mock_request, mock_user, mock_limits):
        """Test that workflow creation is blocked at limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.is_payment_exempt_path") as mock_is_exempt, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_workflow_count") as mock_get_count, \
             patch("seer.api.core.middleware.usage_limit.resolve_user_tier") as mock_resolve_tier:

            mock_is_public.return_value = False
            mock_is_exempt.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 10  # At limit
            mock_resolve_tier.return_value = MagicMock(value="free")

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 402

    @pytest.mark.asyncio
    async def test_workflow_creation_unlimited_tier(self, mock_request, mock_user, unlimited_limits):
        """Test that unlimited tier bypasses workflow limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = unlimited_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should pass through without checking count
            call_next.assert_called_once()


# =============================================================================
# Workflow Run Limit Tests
# =============================================================================


@pytest.mark.unit
class TestWorkflowRunLimit:
    """Tests for workflow run limit enforcement."""

    @pytest.mark.asyncio
    async def test_workflow_run_within_limit(self, mock_request, mock_user, mock_limits):
        """Test that workflow run is allowed within limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows/wf_123/run"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_monthly_run_count") as mock_get_count:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 50  # Under limit of 100

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_run_at_limit(self, mock_request, mock_user, mock_limits):
        """Test that workflow run is blocked at limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows/wf_123/run"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_monthly_run_count") as mock_get_count, \
             patch("seer.api.core.middleware.usage_limit.resolve_user_tier") as mock_resolve_tier:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 100  # At limit
            mock_resolve_tier.return_value = MagicMock(value="free")

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 402

    @pytest.mark.asyncio
    async def test_workflow_run_unlimited_tier(self, mock_request, mock_user, unlimited_limits):
        """Test that unlimited tier bypasses run limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows/wf_123/run"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = unlimited_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()


# =============================================================================
# Polling Interval Validation Tests
# =============================================================================


@pytest.mark.unit
class TestPollingIntervalValidation:
    """Tests for polling interval validation."""

    @pytest.mark.asyncio
    async def test_polling_interval_valid(self, mock_request, mock_user, mock_limits):
        """Test that valid polling interval passes through."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/trigger-subscriptions"
        mock_request.body = AsyncMock(return_value=b'{"poll_interval_seconds": 120}')
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_polling_interval_too_fast_logs_warning(self, mock_request, mock_user, mock_limits):
        """Test that too-fast polling interval logs warning but passes through."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/trigger-subscriptions"
        mock_request.body = AsyncMock(return_value=b'{"poll_interval_seconds": 10}')  # Under min of 60
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.resolve_user_tier") as mock_resolve_tier, \
             patch("seer.api.core.middleware.usage_limit.logger") as mock_logger:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_resolve_tier.return_value = MagicMock(value="free")

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should still pass through (soft enforcement)
            call_next.assert_called_once()
            # Should have logged warning
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_polling_interval_invalid_json_passes_through(self, mock_request, mock_user, mock_limits):
        """Test that invalid JSON in body passes through."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/trigger-subscriptions"
        mock_request.body = AsyncMock(return_value=b'invalid json')
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should pass through even with invalid JSON
            call_next.assert_called_once()


# =============================================================================
# Chat LLM Credit Limit Tests
# =============================================================================


@pytest.mark.unit
class TestChatLLMCreditLimit:
    """Tests for chat LLM credit limit enforcement."""

    @pytest.mark.asyncio
    async def test_chat_endpoint_allowed_when_under_limit(self, mock_request, mock_user, mock_limits):
        """Test that chat requests are allowed when under credit limit."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/nexus/wf_123/chat"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.check_credit_limit") as mock_check_credit:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            # check_credit_limit returns None when under limit (no exception)

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()
            # check_credit_limit now takes (user, organization) - organization is None from mock_request
            mock_check_credit.assert_called_once_with(mock_user, None)

    @pytest.mark.asyncio
    async def test_chat_endpoint_blocked_when_over_limit(self, mock_request, mock_user, mock_limits):
        """Test that chat requests are blocked when credit limit exceeded."""
        from seer.observability.exceptions import CreditLimitExceeded, LimitPeriod
        from seer.database.subscription_models import SubscriptionTier

        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/nexus/wf_123/chat"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.check_credit_limit") as mock_check_credit:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            # Simulate credit limit exceeded
            mock_check_credit.side_effect = CreditLimitExceeded(
                limit=0.02,
                current=0.04,
                tier=SubscriptionTier.PRO,
                period=LimitPeriod.MONTHLY,
            )

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 402
            import json
            body = json.loads(result.body.decode())
            assert body["error"] == "usage_limit_exceeded"
            assert body["resource"] == "llm_credits"
            assert body["period"] == "monthly"

    @pytest.mark.asyncio
    async def test_chat_resume_endpoint_blocked_when_over_limit(self, mock_request, mock_user, mock_limits):
        """Test that chat resume requests are also blocked when credit limit exceeded."""
        from seer.observability.exceptions import CreditLimitExceeded, LimitPeriod
        from seer.database.subscription_models import SubscriptionTier

        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/nexus/wf_123/chat/resume"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.check_credit_limit") as mock_check_credit:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_check_credit.side_effect = CreditLimitExceeded(
                limit=5.00,
                current=6.50,
                tier=SubscriptionTier.PRO,
                period=LimitPeriod.WEEKLY,
            )

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 402

    @pytest.mark.asyncio
    async def test_chat_status_endpoint_not_blocked(self, mock_request, mock_user, mock_limits):
        """Test that GET status endpoint is not subject to credit limit checks."""
        mock_request.state.db_user = mock_user
        mock_request.method = "GET"  # GET requests don't trigger the check
        mock_request.url.path = "/nexus/wf_123/chat/status/123"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.check_credit_limit") as mock_check_credit:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            # Should pass through - GET requests don't trigger POST path checks
            call_next.assert_called_once()
            # Credit check should NOT be called for GET requests
            mock_check_credit.assert_not_called()


# =============================================================================
# Path Pattern Matching Tests
# =============================================================================


@pytest.mark.unit
class TestPathPatternMatching:
    """Tests for path pattern matching in dispatch."""

    @pytest.mark.asyncio
    async def test_get_request_not_checked(self, mock_request, mock_user, mock_limits):
        """Test that GET requests bypass limit checks."""
        mock_request.state.db_user = mock_user
        mock_request.method = "GET"
        mock_request.url.path = "/api/v1/workflows"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_other_post_paths_not_checked(self, mock_request, mock_user, mock_limits):
        """Test that other POST paths bypass limit checks."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/users/settings"
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits

            middleware = UsageLimitMiddleware(app=MagicMock())

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()


# =============================================================================
# Error Response Format Tests
# =============================================================================


@pytest.mark.unit
class TestErrorResponseFormat:
    """Tests for error response format."""

    @pytest.mark.asyncio
    async def test_workflow_limit_error_format(self, mock_request, mock_user, mock_limits):
        """Test that workflow limit error has correct format."""
        mock_request.state.db_user = mock_user
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/workflows"

        with patch("seer.api.core.middleware.usage_limit.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.usage_limit.get_effective_limits") as mock_get_limits, \
             patch("seer.api.core.middleware.usage_limit.get_workflow_count") as mock_get_count, \
             patch("seer.api.core.middleware.usage_limit.resolve_user_tier") as mock_resolve_tier:

            mock_is_public.return_value = False
            mock_get_limits.return_value = mock_limits
            mock_get_count.return_value = 10  # At limit
            mock_resolve_tier.return_value = MagicMock(value="free")

            middleware = UsageLimitMiddleware(app=MagicMock())

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 402
            # Response body should contain error details
            import json
            body = json.loads(result.body.decode())
            assert "error" in body
            assert body["error"] == "usage_limit_exceeded"
