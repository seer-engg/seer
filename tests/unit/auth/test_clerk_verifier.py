"""
Unit tests for ClerkJWTVerifier and ClerkOpaqueTokenVerifier.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from jwt.exceptions import InvalidTokenError

from seer.auth.clerk_verifier import ClerkJWTVerifier, ClerkOpaqueTokenVerifier, VerifiedClerkToken


class TestVerifiedClerkToken:
    """Tests for VerifiedClerkToken dataclass."""

    def test_scopes_from_string(self):
        """Test extracting scopes from space-separated string."""
        token = VerifiedClerkToken(
            user_id="user_123",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            claims={"scope": "read write admin"},
        )
        assert token.scopes == ["read", "write", "admin"]

    def test_scopes_from_list(self):
        """Test extracting scopes from list."""
        token = VerifiedClerkToken(
            user_id="user_123",
            email="test@example.com",
            first_name="Test",
            last_name="User",
            claims={"scope": ["read", "write"]},
        )
        assert token.scopes == ["read", "write"]

    def test_scopes_empty(self):
        """Test empty scopes."""
        token = VerifiedClerkToken(
            user_id="user_123",
            email=None,
            first_name=None,
            last_name=None,
            claims={},
        )
        assert token.scopes == []

    def test_scopes_empty_string(self):
        """Test empty scope string."""
        token = VerifiedClerkToken(
            user_id="user_123",
            email=None,
            first_name=None,
            last_name=None,
            claims={"scope": ""},
        )
        assert token.scopes == []


class TestClerkJWTVerifier:
    """Tests for ClerkJWTVerifier."""

    def test_init_requires_jwks_url(self):
        """Test that jwks_url is required."""
        with pytest.raises(ValueError, match="jwks_url is required"):
            ClerkJWTVerifier(jwks_url="", issuer="https://clerk.example.com")

    def test_init_requires_issuer(self):
        """Test that issuer is required."""
        with pytest.raises(ValueError, match="issuer is required"):
            ClerkJWTVerifier(
                jwks_url="https://clerk.example.com/.well-known/jwks.json",
                issuer=""
            )

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_verify_token_success(self, mock_decode, mock_jwks_client):
        """Test successful token verification."""
        # Setup mocks
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {
            "sub": "user_123",
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
        }

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("valid-token")

        assert result is not None
        assert result.user_id == "user_123"
        assert result.email == "test@example.com"
        assert result.first_name == "Test"
        assert result.last_name == "User"

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_verify_token_invalid(self, mock_decode, mock_jwks_client):
        """Test invalid token returns None."""
        mock_signing_key = MagicMock()
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.side_effect = InvalidTokenError("Token expired")

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("expired-token")

        assert result is None

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_verify_token_with_error_success(self, mock_decode, mock_jwks_client):
        """Test verify_token_with_error returns result and None error on success."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {
            "sub": "user_123",
            "email": "test@example.com",
        }

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result, error = verifier.verify_token_with_error("valid-token")

        assert result is not None
        assert result.user_id == "user_123"
        assert error is None

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_verify_token_with_error_failure(self, mock_decode, mock_jwks_client):
        """Test verify_token_with_error returns None and error message on failure."""
        mock_signing_key = MagicMock()
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.side_effect = InvalidTokenError("Signature verification failed")

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result, error = verifier.verify_token_with_error("invalid-token")

        assert result is None
        assert "Signature verification failed" in error

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_extract_user_id_from_sub(self, mock_decode, mock_jwks_client):
        """Test user_id is extracted from 'sub' claim."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"sub": "user_from_sub"}

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("token")
        assert result.user_id == "user_from_sub"

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_extract_user_id_from_user_id(self, mock_decode, mock_jwks_client):
        """Test user_id is extracted from 'user_id' claim when 'sub' is missing."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"user_id": "user_from_user_id"}

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("token")
        assert result.user_id == "user_from_user_id"

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_extract_user_id_from_sid(self, mock_decode, mock_jwks_client):
        """Test user_id is extracted from 'sid' claim as fallback."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"sid": "session_123"}

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("token")
        assert result.user_id == "session_123"

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_missing_user_id_returns_none(self, mock_decode, mock_jwks_client):
        """Test that missing user identifier returns None."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"email": "test@example.com"}  # No sub, user_id, or sid

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com"
        )

        result = verifier.verify_token("token")
        assert result is None

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_audience_validation(self, mock_decode, mock_jwks_client):
        """Test that audience is passed to jwt.decode."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"sub": "user_123"}

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com",
            audience=["api.example.com", "app.example.com"]
        )

        verifier.verify_token("token")

        # Check that jwt.decode was called with correct audience
        call_kwargs = mock_decode.call_args[1]
        assert call_kwargs["audience"] == ["api.example.com", "app.example.com"]
        assert call_kwargs["options"]["verify_aud"] is True

    @patch("seer.auth.clerk_verifier.PyJWKClient")
    @patch("seer.auth.clerk_verifier.jwt.decode")
    def test_no_audience_skips_validation(self, mock_decode, mock_jwks_client):
        """Test that audience validation is skipped when not configured."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "test-key"
        mock_jwks_client.return_value.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_decode.return_value = {"sub": "user_123"}

        verifier = ClerkJWTVerifier(
            jwks_url="https://clerk.example.com/.well-known/jwks.json",
            issuer="https://clerk.example.com",
            audience=None
        )

        verifier.verify_token("token")

        # Check that audience verification is disabled
        call_kwargs = mock_decode.call_args[1]
        assert call_kwargs["audience"] is None
        assert call_kwargs["options"]["verify_aud"] is False


class TestClerkOpaqueTokenVerifier:
    """Tests for ClerkOpaqueTokenVerifier."""

    def test_init_requires_userinfo_url(self):
        """Test that userinfo_url is required."""
        with pytest.raises(ValueError, match="userinfo_url is required"):
            ClerkOpaqueTokenVerifier(userinfo_url="")

    def test_init_accepts_custom_timeout(self):
        """Test custom timeout parameter."""
        verifier = ClerkOpaqueTokenVerifier(
            userinfo_url="https://clerk.example.com/oauth/userinfo",
            timeout=5.0,
        )
        assert verifier._timeout == 5.0

    @pytest.mark.asyncio
    async def test_verify_valid_token(self):
        """Test successful token verification."""
        mock_userinfo = {
            "user_id": "user_opaque_123",
            "email": "opaque@example.com",
            "email_verified": True,
            "given_name": "Opaque",
            "family_name": "User",
            "picture": "https://example.com/photo.jpg",
        }

        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_userinfo
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result = await verifier.verify_token("valid-opaque-token")

            assert result is not None
            assert result.user_id == "user_opaque_123"
            assert result.email == "opaque@example.com"
            assert result.first_name == "Opaque"
            assert result.last_name == "User"
            assert result.claims["picture"] == "https://example.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_verify_token_with_sub_claim(self):
        """Test that 'sub' claim is used when 'user_id' is missing."""
        mock_userinfo = {
            "sub": "user_from_sub",
            "email": "test@example.com",
        }

        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_userinfo
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result = await verifier.verify_token("token")
            assert result.user_id == "user_from_sub"

    @pytest.mark.asyncio
    async def test_verify_invalid_token_401(self):
        """Test handling of 401 response (invalid/expired token)."""
        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 401

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result, error = await verifier.verify_token_with_error("invalid-token")

            assert result is None
            assert "invalid or expired" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_forbidden_403(self):
        """Test handling of 403 response (insufficient permissions)."""
        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 403

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result, error = await verifier.verify_token_with_error("forbidden-token")

            assert result is None
            assert "insufficient permissions" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_timeout(self):
        """Test handling of timeout."""
        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.TimeoutException("Connection timed out")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result, error = await verifier.verify_token_with_error("some-token")

            assert result is None
            assert "timed out" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_network_error(self):
        """Test handling of network errors."""
        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result, error = await verifier.verify_token_with_error("some-token")

            assert result is None
            assert error is not None
            assert "error" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_missing_user_id(self):
        """Test handling of response missing user_id."""
        mock_userinfo = {
            "email": "test@example.com",
            # No user_id or sub
        }

        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_userinfo
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            result, error = await verifier.verify_token_with_error("token")

            assert result is None
            assert error is not None

    @pytest.mark.asyncio
    async def test_verify_sends_bearer_token(self):
        """Test that Bearer token is sent in Authorization header."""
        mock_userinfo = {"user_id": "user_123"}

        with patch("seer.auth.clerk_verifier.httpx.AsyncClient") as mock_client_class:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_userinfo
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            verifier = ClerkOpaqueTokenVerifier(
                userinfo_url="https://clerk.example.com/oauth/userinfo"
            )

            await verifier.verify_token("my-opaque-token")

            # Verify the GET request was made with correct headers
            mock_client.get.assert_called_once_with(
                "https://clerk.example.com/oauth/userinfo",
                headers={"Authorization": "Bearer my-opaque-token"},
            )
