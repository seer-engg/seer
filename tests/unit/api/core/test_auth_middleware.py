"""
Unit tests for Clerk authentication middleware.

Tests:
- ClerkAuthMiddleware.dispatch: JWT verification flow
- _verify_token: Token validation
- _create_auth_user: User construction from claims
- _check_access_gates: Payment gates
- Token extraction from headers and query params
- Public path handling
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.core.middleware.auth import AuthenticatedUser, ClerkAuthMiddleware
from seer.auth.clerk_verifier import VerifiedClerkToken


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = MagicMock()
    request.method = "GET"
    request.url.path = "/api/v1/workflows"
    request.scope = {"path": "/api/v1/workflows"}
    request.headers = {}
    request.query_params = {}
    request.state = MagicMock()
    return request


@pytest.fixture
def mock_verifier():
    """Create a mock ClerkJWTVerifier."""
    verifier = MagicMock()
    return verifier


@pytest.fixture
def valid_claims():
    """Sample valid JWT claims."""
    return {
        "sub": "user_123",
        "email": "test@example.com",
        "first_name": "Test",
        "last_name": "User",
        "iss": "https://clerk.example.com",
        "aud": ["test_audience"],
        "exp": 9999999999,
        "iat": 1000000000,
    }


@pytest.fixture
def verified_token(valid_claims):
    """Sample VerifiedClerkToken."""
    return VerifiedClerkToken(
        user_id="user_123",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        claims=valid_claims,
    )


@pytest.fixture
def mock_db_user():
    """Create a mock database user."""
    from seer.database import User
    user = MagicMock(spec=User)
    user.id = 1
    user.user_id = "user_123"
    user.email = "test@example.com"
    return user


# =============================================================================
# AuthenticatedUser Tests
# =============================================================================


@pytest.mark.unit
class TestAuthenticatedUser:
    """Tests for AuthenticatedUser dataclass."""

    def test_authenticated_user_creation(self, valid_claims):
        """Test creating AuthenticatedUser from claims."""
        user = AuthenticatedUser(
            user_id="user_123",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            claims=valid_claims,
        )

        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "User"

    def test_authenticated_user_optional_fields(self, valid_claims):
        """Test AuthenticatedUser with optional fields None."""
        user = AuthenticatedUser(
            user_id="user_123",
            email="test@example.com",
            first_name=None,
            last_name=None,
            claims=valid_claims,
        )

        assert user.first_name is None
        assert user.last_name is None

    def test_from_verified_token(self, verified_token):
        """Test creating AuthenticatedUser from VerifiedClerkToken."""
        user = AuthenticatedUser.from_verified_token(verified_token)

        assert user.user_id == "user_123"
        assert user.email == "test@example.com"
        assert user.first_name == "Test"
        assert user.last_name == "User"


# =============================================================================
# ClerkAuthMiddleware Initialization Tests
# =============================================================================


@pytest.mark.unit
class TestClerkAuthMiddlewareInit:
    """Tests for ClerkAuthMiddleware initialization."""

    def test_init_requires_jwks_url(self):
        """Test that initialization requires jwks_url."""
        with pytest.raises(ValueError) as exc_info:
            ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="",
                issuer="https://clerk.example.com",
            )

        assert "jwks_url is required" in str(exc_info.value)

    def test_init_requires_issuer(self):
        """Test that initialization requires issuer."""
        with pytest.raises(ValueError) as exc_info:
            ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="",
            )

        assert "issuer is required" in str(exc_info.value)

    def test_init_with_valid_params(self):
        """Test successful initialization with valid params."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
                audience=["test_audience"],
            )

            # Middleware should have a verifier instance
            assert middleware._verifier is not None


# =============================================================================
# Extract Token Tests
# =============================================================================


@pytest.mark.unit
class TestExtractToken:
    """Tests for token extraction from request."""

    def test_extract_token_from_authorization_header(self, mock_request):
        """Test extracting token from Authorization header."""
        mock_request.headers = {"Authorization": "Bearer valid_token_123"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            token = middleware._extract_token(mock_request)

            assert token == "valid_token_123"

    def test_extract_token_from_query_param(self, mock_request):
        """Test extracting token from query parameter."""
        mock_request.headers = {}
        mock_request.query_params = {"token": "query_token_456"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            token = middleware._extract_token(mock_request)

            assert token == "query_token_456"

    def test_extract_token_prefers_header_over_query(self, mock_request):
        """Test that Authorization header is preferred over query param."""
        mock_request.headers = {"Authorization": "Bearer header_token"}
        mock_request.query_params = {"token": "query_token"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            token = middleware._extract_token(mock_request)

            assert token == "header_token"

    def test_extract_token_no_token_returns_none(self, mock_request):
        """Test that missing token returns None."""
        mock_request.headers = {}
        mock_request.query_params = {}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            token = middleware._extract_token(mock_request)

            assert token is None

    def test_extract_token_invalid_bearer_format(self, mock_request):
        """Test handling invalid Bearer format."""
        mock_request.headers = {"Authorization": "Basic invalid_token"}
        mock_request.query_params = {}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            token = middleware._extract_token(mock_request)

            assert token is None


# =============================================================================
# Should Skip Tests
# =============================================================================


@pytest.mark.unit
class TestShouldSkip:
    """Tests for request skip logic."""

    def test_should_skip_options_request(self, mock_request):
        """Test that OPTIONS requests are skipped."""
        mock_request.method = "OPTIONS"

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            assert middleware._should_skip(mock_request) is True

    def test_should_skip_public_path(self, mock_request):
        """Test that public paths are skipped."""
        mock_request.method = "GET"
        mock_request.scope = {"path": "/health"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.api.core.middleware.auth.is_public_path") as mock_is_public:
            mock_is_public.return_value = True

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            assert middleware._should_skip(mock_request) is True

    def test_should_not_skip_protected_path(self, mock_request):
        """Test that protected paths are not skipped."""
        mock_request.method = "GET"
        mock_request.scope = {"path": "/api/v1/workflows"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.api.core.middleware.auth.is_public_path") as mock_is_public:
            mock_is_public.return_value = False

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            assert middleware._should_skip(mock_request) is False


# =============================================================================
# Create Auth User Tests
# =============================================================================


@pytest.mark.unit
class TestCreateAuthUser:
    """Tests for _create_auth_user method."""

    def test_create_auth_user_from_verified_token(self, verified_token):
        """Test creating auth user from VerifiedClerkToken."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            user = middleware._create_auth_user(verified_token)

            assert user.user_id == "user_123"
            assert user.email == "test@example.com"
            assert user.first_name == "Test"
            assert user.last_name == "User"

    def test_create_auth_user_minimal_token(self):
        """Test creating auth user from minimal VerifiedClerkToken."""
        minimal_token = VerifiedClerkToken(
            user_id="user_minimal",
            email=None,
            first_name=None,
            last_name=None,
            claims={"sub": "user_minimal"},
        )

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            user = middleware._create_auth_user(minimal_token)

            assert user.user_id == "user_minimal"
            assert user.email is None
            assert user.first_name is None
            assert user.last_name is None


# =============================================================================
# Verify Token Tests
# =============================================================================


@pytest.mark.unit
class TestVerifyToken:
    """Tests for _verify_token method."""

    def test_verify_token_success(self, verified_token):
        """Test successful token verification."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            # Mock the verifier to return success
            middleware._verifier.verify_token_with_error = MagicMock(
                return_value=(verified_token, None)
            )

            result = middleware._verify_token("valid_token")

            assert isinstance(result, VerifiedClerkToken)
            assert result.user_id == "user_123"

    def test_verify_token_invalid_token_error(self):
        """Test handling of invalid token."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            # Mock the verifier to return error
            middleware._verifier.verify_token_with_error = MagicMock(
                return_value=(None, "Token expired")
            )

            result = middleware._verify_token("expired_token")

            # Should return JSONResponse
            assert result.status_code == 401

    def test_verify_token_generic_error(self):
        """Test handling of generic verification error."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            # Mock the verifier to return error with None message
            middleware._verifier.verify_token_with_error = MagicMock(
                return_value=(None, None)
            )

            result = middleware._verify_token("token")

            assert result.status_code == 401
            assert b"Authentication failed" in result.body


# =============================================================================
# Check Access Gates Tests
# =============================================================================


@pytest.mark.unit
class TestCheckAccessGates:
    """Tests for _check_access_gates method."""

    @pytest.mark.asyncio
    async def test_check_access_gates_allows_authenticated_user_without_payment_block(self, mock_request, mock_db_user):
        """Test that account age no longer blocks access."""
        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )
            middleware._check_payment_method_gate = AsyncMock(return_value=None)

            result = await middleware._check_access_gates(mock_request, mock_db_user)

            assert result is None
            middleware._check_payment_method_gate.assert_awaited_once_with(mock_db_user)

    @pytest.mark.asyncio
    async def test_payment_exempt_path_skips_payment_gate(self, mock_db_user):
        """Test that payment-exempt paths skip payment gate checks."""
        # Setup request for payment-exempt path
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/subscriptions/checkout"
        request.scope = {"path": "/api/subscriptions/checkout"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )
            middleware._check_payment_method_gate = AsyncMock(return_value=MagicMock())

            result = await middleware._check_access_gates(request, mock_db_user)

            # Should return None (no gate applied)
            assert result is None
            middleware._check_payment_method_gate.assert_not_called()

    @pytest.mark.asyncio
    async def test_payment_exempt_path_skips_payment_method_gate(self, mock_db_user):
        """Test that payment-exempt paths skip payment method gate."""
        # Setup request for payment-exempt path
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/usage"
        request.scope = {"path": "/api/usage"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.database.organization_models.Organization.get_or_none") as mock_billing:
            # User has no payment method
            mock_billing.return_value = None

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            result = await middleware._check_access_gates(request, mock_db_user)

            # Should return None (no gate applied) because this is a payment-exempt path
            assert result is None
            # Payment method check should not even be called for exempt paths
            mock_billing.assert_not_called()

    @pytest.mark.asyncio
    async def test_usage_analytics_prefix_is_payment_exempt(self, mock_db_user):
        """Test that /api/usage/analytics/* paths are payment-exempt."""
        # Setup request for analytics path
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/usage/analytics/daily"
        request.scope = {"path": "/api/usage/analytics/daily"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )
            middleware._check_payment_method_gate = AsyncMock(return_value=MagicMock())

            result = await middleware._check_access_gates(request, mock_db_user)

            # Should skip gate
            assert result is None
            middleware._check_payment_method_gate.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_payment_exempt_path_enforces_gates(self, mock_db_user):
        """Test that non-payment-exempt paths still enforce payment gate."""
        # Setup request for regular workflow path
        request = MagicMock()
        request.method = "GET"
        request.url.path = "/api/v1/workflows"
        request.scope = {"path": "/api/v1/workflows"}

        with patch("seer.auth.clerk_verifier.PyJWKClient"):
            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )
            middleware._check_payment_method_gate = AsyncMock(
                return_value=MagicMock(status_code=402)
            )

            result = await middleware._check_access_gates(request, mock_db_user)

            # Should return 402 error
            assert result is not None
            assert result.status_code == 402
            middleware._check_payment_method_gate.assert_awaited_once_with(mock_db_user)


# =============================================================================
# Dispatch Tests
# =============================================================================


@pytest.mark.unit
class TestDispatch:
    """Tests for dispatch method."""

    @pytest.mark.asyncio
    async def test_dispatch_skips_public_path(self, mock_request):
        """Test that public paths are passed through."""
        mock_request.method = "GET"
        mock_request.scope = {"path": "/health"}

        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.api.core.middleware.auth.is_public_path") as mock_is_public:
            mock_is_public.return_value = True

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            await middleware.dispatch(mock_request, call_next)

            call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_missing_token_returns_401(self, mock_request):
        """Test that missing token returns 401."""
        mock_request.headers = {}
        mock_request.query_params = {}

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.api.core.middleware.auth.is_public_path") as mock_is_public:
            mock_is_public.return_value = False

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            result = await middleware.dispatch(mock_request, AsyncMock())

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_dispatch_sets_user_on_success(self, mock_request, verified_token, mock_db_user):
        """Test that successful auth sets user on request state."""
        mock_request.headers = {"Authorization": "Bearer valid_token"}
        call_next = AsyncMock(return_value=MagicMock())

        with patch("seer.auth.clerk_verifier.PyJWKClient"), \
             patch("seer.api.core.middleware.auth.is_public_path") as mock_is_public, \
             patch("seer.api.core.middleware.auth.User") as MockUser:

            mock_is_public.return_value = False
            MockUser.get_or_create_from_auth = AsyncMock(return_value=mock_db_user)

            middleware = ClerkAuthMiddleware(
                app=MagicMock(),
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer="https://clerk.example.com",
            )

            # Mock the verifier to return success
            middleware._verifier.verify_token_with_error = MagicMock(
                return_value=(verified_token, None)
            )
            # Mock payment method gate to skip it
            middleware._check_payment_method_gate = AsyncMock(return_value=None)

            await middleware.dispatch(mock_request, call_next)

            assert mock_request.state.user is not None
            assert mock_request.state.db_user == mock_db_user
