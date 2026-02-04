"""
Unit tests for tools.credential_resolver module.

Tests the CredentialResolver class for resolving OAuth, resources, and secrets.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock()
    user.user_id = "test_user_123"
    return user


@pytest.fixture
def mock_tool_no_scopes():
    """Create a mock tool with no required scopes."""
    tool = MagicMock()
    tool.name = "test_tool"
    tool.required_scopes = []
    tool.required_secrets = []
    tool.default_resource = None
    tool.provider = None
    tool.integration_type = None
    return tool


@pytest.fixture
def mock_tool_with_scopes():
    """Create a mock tool with required scopes."""
    tool = MagicMock()
    tool.name = "gmail_send"
    tool.required_scopes = ["gmail.send", "gmail.compose"]
    tool.required_secrets = []
    tool.default_resource = None
    tool.provider = "google"
    tool.integration_type = "gmail"
    return tool


@pytest.fixture
def mock_tool_with_secrets():
    """Create a mock tool with required secrets."""
    tool = MagicMock()
    tool.name = "api_tool"
    tool.required_scopes = []
    tool.required_secrets = ["api_key", "api_secret"]
    tool.default_resource = None
    tool.provider = "custom"
    tool.integration_type = "custom"
    return tool


@pytest.fixture
def mock_connection():
    """Create a mock OAuth connection."""
    connection = MagicMock()
    connection.id = 1
    connection.provider = "google"
    connection.scopes = ["gmail.send", "gmail.compose", "gmail.readonly"]
    return connection


@pytest.fixture
def mock_resource():
    """Create a mock integration resource."""
    resource = MagicMock()
    resource.id = 1
    resource.provider = "google"
    resource.oauth_connection_id = 1
    return resource


# =============================================================================
# Resolve Tests - No Scopes Required
# =============================================================================


@pytest.mark.unit
class TestResolveNoScopes:
    """Tests for resolve() when no scopes are required."""

    @pytest.mark.asyncio
    async def test_resolve_no_scopes_returns_empty_credentials(self, mock_user, mock_tool_no_scopes):
        """Test resolve returns empty credentials when no scopes required."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = await resolver.resolve({})

        assert result.connection is None
        assert result.access_token is None
        assert result.resource is None
        assert result.secrets == {}

    @pytest.mark.asyncio
    async def test_resolve_no_scopes_with_default_resource(self, mock_user, mock_tool_no_scopes):
        """Test resolve with default_resource but no scopes."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool_no_scopes.default_resource = {"resource_type": "spreadsheet", "provider": "google"}

        with patch.object(CredentialResolver, "_find_default_resource", new_callable=AsyncMock, return_value=None):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
            result = await resolver.resolve({})

        assert result.connection is None
        assert result.resource is None


# =============================================================================
# Resolve Connection Tests
# =============================================================================


@pytest.mark.unit
class TestResolveConnection:
    """Tests for _resolve_connection method."""

    @pytest.mark.asyncio
    async def test_resolve_connection_no_scopes_returns_none(self, mock_user, mock_tool_no_scopes):
        """Test _resolve_connection returns None when no scopes required."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        connection, token = await resolver._resolve_connection()

        assert connection is None
        assert token is None

    @pytest.mark.asyncio
    async def test_resolve_connection_missing_user_id_raises_401(self, mock_tool_with_scopes):
        """Test _resolve_connection raises 401 when user_id missing."""
        from seer.tools.credential_resolver import CredentialResolver

        user = MagicMock()
        user.user_id = None  # No user ID

        resolver = CredentialResolver(user=user, tool=mock_tool_with_scopes)

        with pytest.raises(HTTPException) as exc_info:
            await resolver._resolve_connection()

        assert exc_info.value.status_code == 401
        assert "User ID is required" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_connection_no_provider_no_connection_id_raises_400(self, mock_user, mock_tool_with_scopes):
        """Test _resolve_connection raises 400 when no provider and no connection_id."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool_with_scopes.provider = None
        mock_tool_with_scopes.integration_type = None

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_scopes, connection_id=None)

        with pytest.raises(HTTPException) as exc_info:
            await resolver._resolve_connection()

        assert exc_info.value.status_code == 400
        assert "connection_id must be provided" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_connection_with_connection_id_success(self, mock_user, mock_tool_with_scopes, mock_connection):
        """Test _resolve_connection succeeds with connection_id."""
        from seer.tools.credential_resolver import CredentialResolver

        with patch("seer.tools.credential_resolver.get_oauth_token", new_callable=AsyncMock, return_value=(mock_connection, "access_token_123")):
            with patch("seer.tools.credential_resolver.validate_scopes", return_value=(True, None)):
                resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_scopes, connection_id="conn_123")
                connection, token = await resolver._resolve_connection()

        assert connection == mock_connection
        assert token == "access_token_123"

    @pytest.mark.asyncio
    async def test_resolve_connection_missing_scope_raises_403(self, mock_user, mock_tool_with_scopes, mock_connection):
        """Test _resolve_connection raises 403 when connection missing required scope."""
        from seer.tools.credential_resolver import CredentialResolver

        with patch("seer.tools.credential_resolver.get_oauth_token", new_callable=AsyncMock, return_value=(mock_connection, "token")):
            with patch("seer.tools.credential_resolver.validate_scopes", return_value=(False, "gmail.admin")):
                resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_scopes, connection_id="conn_123")

                with pytest.raises(HTTPException) as exc_info:
                    await resolver._resolve_connection()

        assert exc_info.value.status_code == 403
        assert "missing required scope" in exc_info.value.detail


# =============================================================================
# Resolve Resource Tests
# =============================================================================


@pytest.mark.unit
class TestResolveResource:
    """Tests for _resolve_resource method."""

    @pytest.mark.asyncio
    async def test_resolve_resource_with_explicit_resource_id(self, mock_user, mock_tool_no_scopes, mock_resource):
        """Test _resolve_resource with explicit resource_id in arguments."""
        from seer.tools.credential_resolver import CredentialResolver

        # Set provider to None to avoid provider mismatch (no connection means no expected provider)
        mock_resource.provider = None

        with patch("seer.tools.credential_resolver.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=mock_resource)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
            result = await resolver._resolve_resource({"resource_id": 123}, None)

        assert result == mock_resource
        MockResource.get_or_none.assert_called_once_with(id=123, user=mock_user, status="active")

    @pytest.mark.asyncio
    async def test_resolve_resource_not_found_raises_404(self, mock_user, mock_tool_no_scopes):
        """Test _resolve_resource raises 404 when resource not found."""
        from seer.tools.credential_resolver import CredentialResolver

        with patch("seer.tools.credential_resolver.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=None)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)

            with pytest.raises(HTTPException) as exc_info:
                await resolver._resolve_resource({"integration_resource_id": 999}, None)

        assert exc_info.value.status_code == 404
        assert "not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_resource_provider_mismatch_raises_400(self, mock_user, mock_tool_no_scopes, mock_resource, mock_connection):
        """Test _resolve_resource raises 400 on provider mismatch."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_resource.provider = "github"  # Different from connection's google
        mock_connection.provider = "google"

        with patch("seer.tools.credential_resolver.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=mock_resource)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)

            with pytest.raises(HTTPException) as exc_info:
                await resolver._resolve_resource({"resource_id": 1}, mock_connection)

        assert exc_info.value.status_code == 400
        assert "provider mismatch" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_resource_connection_mismatch_raises_400(self, mock_user, mock_tool_no_scopes, mock_resource, mock_connection):
        """Test _resolve_resource raises 400 when resource connection doesn't match."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_resource.oauth_connection_id = 999  # Different from connection.id
        mock_connection.id = 1

        with patch("seer.tools.credential_resolver.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=mock_resource)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)

            with pytest.raises(HTTPException) as exc_info:
                await resolver._resolve_resource({"resource_id": 1}, mock_connection)

        assert exc_info.value.status_code == 400
        assert "does not belong" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_resource_no_resource_id_no_default(self, mock_user, mock_tool_no_scopes):
        """Test _resolve_resource returns None when no resource_id and no default."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = await resolver._resolve_resource({}, None)

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_resource_default_required_but_not_found_raises_400(self, mock_user, mock_tool_no_scopes):
        """Test _resolve_resource raises 400 when required default resource not found."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool_no_scopes.default_resource = {"resource_type": "spreadsheet", "required": True}

        with patch.object(CredentialResolver, "_find_default_resource", new_callable=AsyncMock, return_value=None):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)

            with pytest.raises(HTTPException) as exc_info:
                await resolver._resolve_resource({}, None)

        assert exc_info.value.status_code == 400
        assert "requires a persisted" in exc_info.value.detail


# =============================================================================
# Resolve Secrets Tests
# =============================================================================


@pytest.mark.unit
class TestResolveSecrets:
    """Tests for _resolve_secrets method."""

    @pytest.mark.asyncio
    async def test_resolve_secrets_no_secrets_required(self, mock_user, mock_tool_no_scopes):
        """Test _resolve_secrets returns empty when no secrets required."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        secrets, records = await resolver._resolve_secrets(None, None)

        assert secrets == {}
        assert records == {}

    @pytest.mark.asyncio
    async def test_resolve_secrets_no_provider_raises_400(self, mock_user, mock_tool_with_secrets):
        """Test _resolve_secrets raises 400 when no provider can be inferred."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool_with_secrets.provider = None
        mock_tool_with_secrets.integration_type = None

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_secrets)

        with pytest.raises(HTTPException) as exc_info:
            await resolver._resolve_secrets(None, None)

        assert exc_info.value.status_code == 400
        assert "no provider could be inferred" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_secrets_missing_secret_raises_404(self, mock_user, mock_tool_with_secrets):
        """Test _resolve_secrets raises 404 when required secret not found."""
        from seer.tools.credential_resolver import CredentialResolver

        with patch.object(CredentialResolver, "_find_secret", new_callable=AsyncMock, return_value=None):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_secrets)

            with pytest.raises(HTTPException) as exc_info:
                await resolver._resolve_secrets(None, None)

        assert exc_info.value.status_code == 404
        assert "Missing required secret" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_resolve_secrets_success(self, mock_user, mock_tool_with_secrets):
        """Test _resolve_secrets returns secrets when found."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_secret = MagicMock()
        mock_secret.value_enc = "encrypted_value"

        with patch.object(CredentialResolver, "_find_secret", new_callable=AsyncMock, return_value=mock_secret):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_secrets)
            secrets, records = await resolver._resolve_secrets(None, None)

        assert "api_key" in secrets
        assert "api_secret" in secrets
        assert secrets["api_key"] == "encrypted_value"
        assert records["api_key"] == mock_secret


# =============================================================================
# Find Secret Tests
# =============================================================================


@pytest.mark.unit
class TestFindSecret:
    """Tests for _find_secret method."""

    @pytest.mark.asyncio
    async def test_find_secret_resource_level_first(self, mock_user, mock_tool_no_scopes, mock_resource):
        """Test _find_secret checks resource-level secrets first."""
        from seer.tools.credential_resolver import CredentialResolver

        resource_secret = MagicMock()
        resource_secret.value_enc = "resource_secret_value"

        with patch("seer.tools.credential_resolver.IntegrationSecret") as MockSecret:
            MockSecret.get_or_none = AsyncMock(return_value=resource_secret)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
            result = await resolver._find_secret("google", "api_key", mock_resource, None)

        assert result == resource_secret
        # Should have been called with resource filter
        MockSecret.get_or_none.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_secret_connection_level_fallback(self, mock_user, mock_tool_no_scopes, mock_connection):
        """Test _find_secret falls back to connection-level secrets."""
        from seer.tools.credential_resolver import CredentialResolver

        connection_secret = MagicMock()
        connection_secret.value_enc = "connection_secret_value"

        with patch("seer.tools.credential_resolver.IntegrationSecret") as MockSecret:
            # First call (resource) returns None, second call (connection) returns secret
            MockSecret.get_or_none = AsyncMock(side_effect=[None, connection_secret])

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
            result = await resolver._find_secret("google", "api_key", None, mock_connection)

        assert result == connection_secret

    @pytest.mark.asyncio
    async def test_find_secret_global_fallback(self, mock_user, mock_tool_no_scopes):
        """Test _find_secret falls back to global secrets."""
        from seer.tools.credential_resolver import CredentialResolver

        global_secret = MagicMock()
        global_secret.value_enc = "global_secret_value"

        with patch("seer.tools.credential_resolver.IntegrationSecret") as MockSecret:
            # No resource or connection secrets, only global
            MockSecret.get_or_none = AsyncMock(return_value=global_secret)

            resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
            result = await resolver._find_secret("google", "api_key", None, None)

        assert result == global_secret


# =============================================================================
# Extract Resource ID Tests
# =============================================================================


@pytest.mark.unit
class TestExtractResourceId:
    """Tests for _extract_resource_id static method."""

    def test_extract_resource_id_from_integration_resource_id(self):
        """Test extracting resource_id from integration_resource_id key."""
        from seer.tools.credential_resolver import CredentialResolver

        result = CredentialResolver._extract_resource_id({"integration_resource_id": 123})
        assert result == 123

    def test_extract_resource_id_from_resource_id(self):
        """Test extracting resource_id from resource_id key."""
        from seer.tools.credential_resolver import CredentialResolver

        result = CredentialResolver._extract_resource_id({"resource_id": "456"})
        assert result == 456

    def test_extract_resource_id_from_resource_binding_id(self):
        """Test extracting resource_id from resource_binding_id key."""
        from seer.tools.credential_resolver import CredentialResolver

        result = CredentialResolver._extract_resource_id({"resource_binding_id": 789})
        assert result == 789

    def test_extract_resource_id_none_when_not_present(self):
        """Test _extract_resource_id returns None when no key present."""
        from seer.tools.credential_resolver import CredentialResolver

        result = CredentialResolver._extract_resource_id({"other_key": "value"})
        assert result is None

    def test_extract_resource_id_invalid_value_raises_400(self):
        """Test _extract_resource_id raises 400 on invalid value."""
        from seer.tools.credential_resolver import CredentialResolver

        with pytest.raises(HTTPException) as exc_info:
            CredentialResolver._extract_resource_id({"resource_id": "not_a_number"})

        assert exc_info.value.status_code == 400
        assert "Invalid resource identifier" in exc_info.value.detail


# =============================================================================
# Infer Provider Tests
# =============================================================================


@pytest.mark.unit
class TestInferProvider:
    """Tests for _infer_provider method."""

    def test_infer_provider_from_resource(self, mock_user, mock_tool_no_scopes, mock_resource):
        """Test _infer_provider returns resource provider first."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_resource.provider = "github"

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = resolver._infer_provider(resource=mock_resource)

        assert result == "github"

    def test_infer_provider_from_connection(self, mock_user, mock_tool_no_scopes, mock_connection):
        """Test _infer_provider returns connection provider when no resource."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_connection.provider = "google"

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = resolver._infer_provider(connection=mock_connection)

        assert result == "google"

    def test_infer_provider_from_tool(self, mock_user, mock_tool_with_scopes):
        """Test _infer_provider returns tool provider when no resource or connection."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_with_scopes)
        result = resolver._infer_provider()

        assert result == "google"

    def test_infer_provider_from_integration_type(self, mock_user, mock_tool_no_scopes):
        """Test _infer_provider returns integration_type when no provider."""
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool_no_scopes.integration_type = "slack"

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = resolver._infer_provider()

        assert result == "slack"

    def test_infer_provider_none_when_nothing_available(self, mock_user, mock_tool_no_scopes):
        """Test _infer_provider returns None when nothing available."""
        from seer.tools.credential_resolver import CredentialResolver

        resolver = CredentialResolver(user=mock_user, tool=mock_tool_no_scopes)
        result = resolver._infer_provider()

        assert result is None
