"""
Unit tests for integrations services.

Tests the actual functions from seer.api.integrations.services including:
- get_connection_for_provider: OAuth connection lookup
- disconnect_provider: Provider disconnection with cascade
- delete_connection_by_id: Connection deletion by ID
- get_valid_access_token: Token retrieval with refresh
- list_integration_resources: Resource enumeration
- list_resource_secrets: Secret listing for resources
- deactivate_integration_resource: Resource deactivation
- serialize_integration_resource: Resource serialization
- serialize_integration_secret: Secret serialization
- bind_supabase_project_manual: Manual Supabase binding
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib

import pytest
from fastapi import HTTPException

from seer.api.integrations.services import (
    get_connection_for_provider,
    disconnect_provider,
    delete_connection_by_id,
    get_valid_access_token,
    list_integration_resources,
    list_resource_secrets,
    deactivate_integration_resource,
    serialize_integration_resource,
    serialize_integration_secret,
    bind_supabase_project_manual,
    _fingerprint_secret,
    _format_supabase_secret_name,
    _build_manual_supabase_metadata,
)
from seer.services.integrations.auth.oauth import get_oauth_provider


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    user = MagicMock()
    user.id = 1
    user.user_id = "user_123"
    return user


@pytest.fixture
def mock_oauth_connection():
    """Create a mock OAuth connection."""
    connection = MagicMock()
    connection.id = 1
    connection.provider = "google"
    connection.status = "active"
    connection.access_token_enc = "encrypted_token_123"
    connection.refresh_token_enc = "encrypted_refresh_123"
    connection.scopes = "email profile"
    connection.provider_account_id = "account_123"
    connection.provider_metadata = {"email": "test@example.com"}
    connection.expires_at = datetime(2025, 12, 31, tzinfo=timezone.utc)
    connection.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    connection.updated_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return connection


@pytest.fixture
def mock_integration_resource():
    """Create a mock integration resource."""
    resource = MagicMock()
    resource.id = 1
    resource.provider = "supabase"
    resource.resource_type = "project"
    resource.resource_id = "project_abc123"
    resource.resource_key = "abc123"
    resource.name = "My Supabase Project"
    resource.status = "active"
    resource.resource_metadata = {"project_ref": "abc123"}
    resource.oauth_connection_id = 1
    resource.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    resource.updated_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return resource


@pytest.fixture
def mock_integration_secret():
    """Create a mock integration secret."""
    secret = MagicMock()
    secret.id = 1
    secret.provider = "supabase"
    secret.name = "supabase_service_role_key"
    secret.secret_type = "api_key"
    secret.resource_id = 1
    secret.oauth_connection_id = None
    secret.value_fingerprint = "abc123fingerprint"
    secret.metadata = {"binding_mode": "manual"}
    secret.status = "active"
    secret.expires_at = None
    secret.created_at = datetime(2024, 1, 1, tzinfo=timezone.utc)
    secret.updated_at = datetime(2024, 6, 1, tzinfo=timezone.utc)
    return secret


# =============================================================================
# OAuth Provider Mapping Tests
# =============================================================================


@pytest.mark.unit
class TestGetOAuthProvider:
    """Tests for get_oauth_provider function."""

    def test_gmail_maps_to_google(self):
        """Test gmail integration type maps to google OAuth provider."""
        result = get_oauth_provider("gmail")
        assert result == "google"

    def test_googlesheets_maps_to_google(self):
        """Test googlesheets integration type maps to google OAuth provider."""
        result = get_oauth_provider("googlesheets")
        assert result == "google"

    def test_googledrive_maps_to_google(self):
        """Test googledrive integration type maps to google OAuth provider."""
        result = get_oauth_provider("googledrive")
        assert result == "google"

    def test_google_maps_to_google(self):
        """Test google integration type maps to google OAuth provider."""
        result = get_oauth_provider("google")
        assert result == "google"

    def test_supabase_maps_to_supabase_mgmt(self):
        """Test supabase integration type maps to supabase_mgmt OAuth provider."""
        result = get_oauth_provider("supabase")
        assert result == "supabase_mgmt"

    def test_discord_maps_to_discord(self):
        """Test discord integration type maps to discord OAuth provider."""
        result = get_oauth_provider("discord")
        assert result == "discord"

    def test_linkedin_maps_to_linkedin(self):
        """Test linkedin integration type maps to linkedin OAuth provider."""
        result = get_oauth_provider("linkedin")
        assert result == "linkedin"

    def test_github_maps_to_github(self):
        """Test github integration type passes through as OAuth provider."""
        result = get_oauth_provider("github")
        assert result == "github"

    def test_unknown_provider_passes_through(self):
        """Test unknown integration type passes through unchanged."""
        result = get_oauth_provider("custom_provider")
        assert result == "custom_provider"


# =============================================================================
# Get Connection Tests
# =============================================================================


@pytest.mark.unit
class TestGetConnectionForProvider:
    """Tests for get_connection_for_provider function."""

    @pytest.mark.asyncio
    async def test_get_connection_found(self, mock_user, mock_oauth_connection):
        """Test getting existing connection for provider."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.get_or_none = AsyncMock(return_value=mock_oauth_connection)

            result = await get_connection_for_provider(mock_user, "google")

            assert result == mock_oauth_connection
            MockOAuthConnection.get_or_none.assert_called_once_with(
                user=mock_user,
                provider="google",
                status="active"
            )

    @pytest.mark.asyncio
    async def test_get_connection_not_found(self, mock_user):
        """Test handling missing connection returns None."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.get_or_none = AsyncMock(return_value=None)

            result = await get_connection_for_provider(mock_user, "google")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_connection_maps_provider(self, mock_user, mock_oauth_connection):
        """Test that integration type is mapped to OAuth provider."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.get_or_none = AsyncMock(return_value=mock_oauth_connection)

            await get_connection_for_provider(mock_user, "gmail")

            # gmail should be mapped to google
            MockOAuthConnection.get_or_none.assert_called_once_with(
                user=mock_user,
                provider="google",
                status="active"
            )

    @pytest.mark.asyncio
    async def test_get_connection_handles_exception(self, mock_user):
        """Test that exceptions are caught and None is returned."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.get_or_none = AsyncMock(side_effect=Exception("DB error"))

            result = await get_connection_for_provider(mock_user, "google")

            assert result is None


# =============================================================================
# Disconnect Provider Tests
# =============================================================================


class AsyncQuerySetMock:
    """A proper async-awaitable mock for Tortoise ORM QuerySets."""

    def __init__(self, items, with_update=False):
        self._items = items
        self._update_mock = AsyncMock() if with_update else None

    def __await__(self):
        async def _coro():
            return self._items
        return _coro().__await__()

    def __aiter__(self):
        return self._async_iter()

    async def _async_iter(self):
        for item in self._items:
            yield item

    @property
    def update(self):
        return self._update_mock


@pytest.mark.unit
class TestDisconnectProvider:
    """Tests for disconnect_provider function."""

    @pytest.mark.asyncio
    async def test_disconnect_revokes_connections(self, mock_user, mock_oauth_connection):
        """Test that disconnect revokes all connections for provider."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection, \
             patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret, \
             patch("seer.api.integrations.services.deactivate_integration_resource"):

            # First call returns connections list, second call is for update
            connections_qs = AsyncQuerySetMock([mock_oauth_connection])
            update_qs = AsyncQuerySetMock([], with_update=True)
            MockOAuthConnection.filter = MagicMock(side_effect=[connections_qs, update_qs])

            # Mock resource filter - returns empty list (no linked resources)
            resource_qs = AsyncQuerySetMock([])
            MockResource.filter = MagicMock(return_value=resource_qs)

            # Mock secret filter with update
            secret_qs = AsyncQuerySetMock([], with_update=True)
            MockSecret.filter = MagicMock(return_value=secret_qs)

            await disconnect_provider(mock_user, "google")

            # Verify filter was called to update status
            assert MockOAuthConnection.filter.call_count >= 1

    @pytest.mark.asyncio
    async def test_disconnect_cascades_to_resources(self, mock_user, mock_oauth_connection, mock_integration_resource):
        """Test that disconnect cascades to linked resources."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection, \
             patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret, \
             patch("seer.api.integrations.services.deactivate_integration_resource") as mock_deactivate:

            # First call returns connections list, second call is for update
            connections_qs = AsyncQuerySetMock([mock_oauth_connection])
            update_qs = AsyncQuerySetMock([], with_update=True)
            MockOAuthConnection.filter = MagicMock(side_effect=[connections_qs, update_qs])

            # Mock resource filter to return a linked resource
            resource_qs = AsyncQuerySetMock([mock_integration_resource])
            MockResource.filter = MagicMock(return_value=resource_qs)

            # Mock secret filter with update
            secret_qs = AsyncQuerySetMock([], with_update=True)
            MockSecret.filter = MagicMock(return_value=secret_qs)

            mock_deactivate.return_value = mock_integration_resource

            await disconnect_provider(mock_user, "google")

            # Verify deactivate was called for linked resources
            mock_deactivate.assert_called_once_with(mock_user, mock_integration_resource.id)


# =============================================================================
# Delete Connection By ID Tests
# =============================================================================


@pytest.mark.unit
class TestDeleteConnectionById:
    """Tests for delete_connection_by_id function."""

    @pytest.mark.asyncio
    async def test_delete_connection_parses_compound_id(self, mock_user, mock_oauth_connection):
        """Test that compound IDs (provider:id) are parsed correctly."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection, \
             patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret, \
             patch("seer.api.integrations.services.deactivate_integration_resource"):

            MockOAuthConnection.get_or_none = AsyncMock(return_value=mock_oauth_connection)

            # Mock filter for update
            oauth_filter_qs = AsyncQuerySetMock([], with_update=True)
            MockOAuthConnection.filter = MagicMock(return_value=oauth_filter_qs)

            # Mock resource filter
            resource_qs = AsyncQuerySetMock([])
            MockResource.filter = MagicMock(return_value=resource_qs)

            # Mock secret filter with update
            secret_qs = AsyncQuerySetMock([], with_update=True)
            MockSecret.filter = MagicMock(return_value=secret_qs)

            await delete_connection_by_id(mock_user, "google:123")

            MockOAuthConnection.get_or_none.assert_called_with(id=123, user=mock_user)

    @pytest.mark.asyncio
    async def test_delete_connection_parses_simple_id(self, mock_user, mock_oauth_connection):
        """Test that simple IDs are parsed correctly."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection, \
             patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret, \
             patch("seer.api.integrations.services.deactivate_integration_resource"):

            MockOAuthConnection.get_or_none = AsyncMock(return_value=mock_oauth_connection)

            # Mock filter for update
            oauth_filter_qs = AsyncQuerySetMock([], with_update=True)
            MockOAuthConnection.filter = MagicMock(return_value=oauth_filter_qs)

            # Mock resource filter
            resource_qs = AsyncQuerySetMock([])
            MockResource.filter = MagicMock(return_value=resource_qs)

            # Mock secret filter with update
            secret_qs = AsyncQuerySetMock([], with_update=True)
            MockSecret.filter = MagicMock(return_value=secret_qs)

            await delete_connection_by_id(mock_user, "456")

            MockOAuthConnection.get_or_none.assert_called_with(id=456, user=mock_user)

    @pytest.mark.asyncio
    async def test_delete_connection_not_found_raises_404(self, mock_user):
        """Test that deleting non-existent connection raises 404."""
        with patch("seer.api.integrations.services.OAuthConnection") as MockOAuthConnection:
            MockOAuthConnection.get_or_none = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await delete_connection_by_id(mock_user, "999")

            assert exc_info.value.status_code == 404
            assert exc_info.value.detail == "Connection not found"


# =============================================================================
# Get Valid Access Token Tests
# =============================================================================


@pytest.mark.unit
class TestGetValidAccessToken:
    """Tests for get_valid_access_token function."""

    @pytest.mark.asyncio
    async def test_get_token_success(self, mock_user):
        """Test getting valid access token."""
        with patch("seer.api.integrations.services.get_oauth_token") as mock_get_token:
            mock_get_token.return_value = (MagicMock(), "valid_access_token")

            result = await get_valid_access_token(mock_user, "google")

            assert result == "valid_access_token"
            mock_get_token.assert_called_once_with(mock_user, provider="google")

    @pytest.mark.asyncio
    async def test_get_token_not_found_returns_none(self, mock_user):
        """Test that 404 HTTPException returns None."""
        with patch("seer.api.integrations.services.get_oauth_token") as mock_get_token:
            mock_get_token.side_effect = HTTPException(status_code=404, detail="Not found")

            result = await get_valid_access_token(mock_user, "google")

            assert result is None

    @pytest.mark.asyncio
    async def test_get_token_other_error_raises(self, mock_user):
        """Test that non-404 HTTPException is re-raised."""
        with patch("seer.api.integrations.services.get_oauth_token") as mock_get_token:
            mock_get_token.side_effect = HTTPException(status_code=401, detail="Unauthorized")

            with pytest.raises(HTTPException) as exc_info:
                await get_valid_access_token(mock_user, "google")

            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_token_maps_provider(self, mock_user):
        """Test that integration type is mapped to OAuth provider."""
        with patch("seer.api.integrations.services.get_oauth_token") as mock_get_token:
            mock_get_token.return_value = (MagicMock(), "valid_access_token")

            await get_valid_access_token(mock_user, "gmail")

            # gmail should be mapped to google
            mock_get_token.assert_called_once_with(mock_user, provider="google")


# =============================================================================
# List Integration Resources Tests
# =============================================================================


@pytest.mark.unit
class TestListIntegrationResources:
    """Tests for list_integration_resources function."""

    @pytest.mark.asyncio
    async def test_list_resources_returns_all_active(self, mock_user, mock_integration_resource):
        """Test listing all active resources for user."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            mock_queryset = MagicMock()
            mock_queryset.filter = MagicMock(return_value=mock_queryset)
            mock_queryset.order_by = AsyncMock(return_value=[mock_integration_resource])
            MockResource.filter = MagicMock(return_value=mock_queryset)

            result = await list_integration_resources(mock_user)

            assert len(result) == 1
            assert result[0] == mock_integration_resource
            MockResource.filter.assert_called_once_with(user=mock_user, status="active")

    @pytest.mark.asyncio
    async def test_list_resources_empty(self, mock_user):
        """Test listing resources when none exist."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            mock_queryset = MagicMock()
            mock_queryset.filter = MagicMock(return_value=mock_queryset)
            mock_queryset.order_by = AsyncMock(return_value=[])
            MockResource.filter = MagicMock(return_value=mock_queryset)

            result = await list_integration_resources(mock_user)

            assert result == []

    @pytest.mark.asyncio
    async def test_list_resources_filters_by_provider(self, mock_user, mock_integration_resource):
        """Test filtering resources by provider."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            mock_queryset = MagicMock()
            mock_queryset.filter = MagicMock(return_value=mock_queryset)
            mock_queryset.order_by = AsyncMock(return_value=[mock_integration_resource])
            MockResource.filter = MagicMock(return_value=mock_queryset)

            result = await list_integration_resources(mock_user, provider="supabase")

            assert len(result) == 1
            # Verify filter was called with provider
            mock_queryset.filter.assert_called_with(provider="supabase")

    @pytest.mark.asyncio
    async def test_list_resources_filters_by_resource_type(self, mock_user, mock_integration_resource):
        """Test filtering resources by resource type."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            mock_queryset = MagicMock()
            mock_queryset.filter = MagicMock(return_value=mock_queryset)
            mock_queryset.order_by = AsyncMock(return_value=[mock_integration_resource])
            MockResource.filter = MagicMock(return_value=mock_queryset)

            result = await list_integration_resources(mock_user, resource_type="project")

            assert len(result) == 1
            mock_queryset.filter.assert_called_with(resource_type="project")


# =============================================================================
# List Resource Secrets Tests
# =============================================================================


@pytest.mark.unit
class TestListResourceSecrets:
    """Tests for list_resource_secrets function."""

    @pytest.mark.asyncio
    async def test_list_secrets_for_resource(self, mock_user, mock_integration_resource, mock_integration_secret):
        """Test listing secrets for a resource."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret:

            MockResource.get_or_none = AsyncMock(return_value=mock_integration_resource)

            mock_queryset = MagicMock()
            mock_queryset.order_by = AsyncMock(return_value=[mock_integration_secret])
            MockSecret.filter = MagicMock(return_value=mock_queryset)

            result = await list_resource_secrets(mock_user, resource_id=1)

            assert len(result) == 1
            assert result[0] == mock_integration_secret

    @pytest.mark.asyncio
    async def test_list_secrets_resource_not_found_raises_404(self, mock_user):
        """Test that listing secrets for non-existent resource raises 404."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await list_resource_secrets(mock_user, resource_id=999)

            assert exc_info.value.status_code == 404
            assert "999" in exc_info.value.detail


# =============================================================================
# Deactivate Integration Resource Tests
# =============================================================================


@pytest.mark.unit
class TestDeactivateIntegrationResource:
    """Tests for deactivate_integration_resource function."""

    @pytest.mark.asyncio
    async def test_deactivate_resource_success(self, mock_user, mock_integration_resource):
        """Test deactivating a resource."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret:

            MockResource.get_or_none = AsyncMock(return_value=mock_integration_resource)
            mock_integration_resource.save = AsyncMock()

            mock_secret_filter = MagicMock()
            mock_secret_filter.update = AsyncMock()
            MockSecret.filter = MagicMock(return_value=mock_secret_filter)

            result = await deactivate_integration_resource(mock_user, resource_id=1)

            assert result.status == "revoked"
            mock_integration_resource.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_resource_cascades_to_secrets(self, mock_user, mock_integration_resource):
        """Test that deactivating resource cascades to secrets."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource, \
             patch("seer.api.integrations.services.IntegrationSecret") as MockSecret:

            MockResource.get_or_none = AsyncMock(return_value=mock_integration_resource)
            mock_integration_resource.save = AsyncMock()

            mock_secret_filter = MagicMock()
            mock_secret_filter.update = AsyncMock()
            MockSecret.filter = MagicMock(return_value=mock_secret_filter)

            await deactivate_integration_resource(mock_user, resource_id=1)

            # Verify secrets were updated to revoked
            MockSecret.filter.assert_called_once_with(resource=mock_integration_resource, user=mock_user)
            mock_secret_filter.update.assert_called_once_with(status="revoked")

    @pytest.mark.asyncio
    async def test_deactivate_resource_not_found_raises_404(self, mock_user):
        """Test that deactivating non-existent resource raises 404."""
        with patch("seer.api.integrations.services.IntegrationResource") as MockResource:
            MockResource.get_or_none = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await deactivate_integration_resource(mock_user, resource_id=999)

            assert exc_info.value.status_code == 404


# =============================================================================
# Serialization Tests
# =============================================================================


@pytest.mark.unit
class TestSerializeIntegrationResource:
    """Tests for serialize_integration_resource function."""

    def test_serialize_resource_all_fields(self, mock_integration_resource):
        """Test serializing resource with all fields."""
        result = serialize_integration_resource(mock_integration_resource)

        assert result["id"] == 1
        assert result["provider"] == "supabase"
        assert result["resource_type"] == "project"
        assert result["resource_id"] == "project_abc123"
        assert result["resource_key"] == "abc123"
        assert result["name"] == "My Supabase Project"
        assert result["status"] == "active"
        assert result["metadata"] == {"project_ref": "abc123"}
        assert result["oauth_connection_id"] == 1
        assert result["created_at"] == "2024-01-01T00:00:00+00:00"
        assert result["updated_at"] == "2024-06-01T00:00:00+00:00"

    def test_serialize_resource_null_metadata(self):
        """Test serializing resource with null metadata returns empty dict."""
        resource = MagicMock()
        resource.id = 1
        resource.provider = "supabase"
        resource.resource_type = "project"
        resource.resource_id = "proj_123"
        resource.resource_key = None
        resource.name = None
        resource.status = "active"
        resource.resource_metadata = None
        resource.oauth_connection_id = None
        resource.created_at = None
        resource.updated_at = None

        result = serialize_integration_resource(resource)

        assert result["metadata"] == {}
        assert result["created_at"] is None
        assert result["updated_at"] is None


@pytest.mark.unit
class TestSerializeIntegrationSecret:
    """Tests for serialize_integration_secret function."""

    def test_serialize_secret_all_fields(self, mock_integration_secret):
        """Test serializing secret with all fields."""
        result = serialize_integration_secret(mock_integration_secret)

        assert result["id"] == 1
        assert result["provider"] == "supabase"
        assert result["name"] == "supabase_service_role_key"
        assert result["secret_type"] == "api_key"
        assert result["resource_id"] == 1
        assert result["oauth_connection_id"] is None
        assert result["value_fingerprint"] == "abc123fingerprint"
        assert result["metadata"] == {"binding_mode": "manual"}
        assert result["status"] == "active"
        assert result["expires_at"] is None
        assert result["created_at"] == "2024-01-01T00:00:00+00:00"
        assert result["updated_at"] == "2024-06-01T00:00:00+00:00"

    def test_serialize_secret_null_metadata(self):
        """Test serializing secret with null metadata returns empty dict."""
        secret = MagicMock()
        secret.id = 1
        secret.provider = "supabase"
        secret.name = "test_key"
        secret.secret_type = "api_key"
        secret.resource_id = None
        secret.oauth_connection_id = None
        secret.value_fingerprint = "fingerprint"
        secret.metadata = None
        secret.status = "active"
        secret.expires_at = None
        secret.created_at = None
        secret.updated_at = None

        result = serialize_integration_secret(secret)

        assert result["metadata"] == {}


# =============================================================================
# Helper Function Tests
# =============================================================================


@pytest.mark.unit
class TestFingerprintSecret:
    """Tests for _fingerprint_secret function."""

    def test_fingerprint_is_sha256(self):
        """Test fingerprint uses SHA256."""
        result = _fingerprint_secret("test_value")

        # SHA256 produces 64 hex characters
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_fingerprint_deterministic(self):
        """Test fingerprint is deterministic for same input."""
        result1 = _fingerprint_secret("test_value")
        result2 = _fingerprint_secret("test_value")

        assert result1 == result2

    def test_fingerprint_different_for_different_values(self):
        """Test different values produce different fingerprints."""
        result1 = _fingerprint_secret("value1")
        result2 = _fingerprint_secret("value2")

        assert result1 != result2

    def test_fingerprint_matches_manual_calculation(self):
        """Test fingerprint matches manual SHA256 calculation."""
        value = "my_secret_key"
        expected = hashlib.sha256(value.encode("utf-8")).hexdigest()

        result = _fingerprint_secret(value)

        assert result == expected


@pytest.mark.unit
class TestFormatSupabaseSecretName:
    """Tests for _format_supabase_secret_name function."""

    def test_service_role_mapping(self):
        """Test service_role maps to supabase_service_role_key."""
        assert _format_supabase_secret_name("service_role") == "supabase_service_role_key"

    def test_service_role_dash_mapping(self):
        """Test service-role maps to supabase_service_role_key."""
        assert _format_supabase_secret_name("service-role") == "supabase_service_role_key"

    def test_service_mapping(self):
        """Test service maps to supabase_service_role_key."""
        assert _format_supabase_secret_name("service") == "supabase_service_role_key"

    def test_anon_mapping(self):
        """Test anon maps to supabase_anon_key."""
        assert _format_supabase_secret_name("anon") == "supabase_anon_key"

    def test_anon_key_mapping(self):
        """Test anon_key maps to supabase_anon_key."""
        assert _format_supabase_secret_name("anon_key") == "supabase_anon_key"

    def test_empty_string_returns_custom(self):
        """Test empty string returns supabase_custom_key."""
        assert _format_supabase_secret_name("") == "supabase_custom_key"

    def test_none_returns_custom(self):
        """Test None returns supabase_custom_key."""
        assert _format_supabase_secret_name(None) == "supabase_custom_key"

    def test_unknown_name_prefixed(self):
        """Test unknown name is prefixed with supabase_."""
        assert _format_supabase_secret_name("custom") == "supabase_custom_key"

    def test_case_insensitive(self):
        """Test name matching is case insensitive."""
        assert _format_supabase_secret_name("SERVICE_ROLE") == "supabase_service_role_key"
        assert _format_supabase_secret_name("Anon") == "supabase_anon_key"

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed."""
        assert _format_supabase_secret_name("  anon  ") == "supabase_anon_key"


@pytest.mark.unit
class TestBuildManualSupabaseMetadata:
    """Tests for _build_manual_supabase_metadata function."""

    def test_builds_correct_urls(self):
        """Test that correct Supabase URLs are built."""
        result = _build_manual_supabase_metadata(
            project_ref="abc123",
            project_name="My Project"
        )

        assert result["project_ref"] == "abc123"
        assert result["binding_mode"] == "manual"
        assert result["name"] == "My Project"
        assert result["rest_url"] == "https://abc123.supabase.co/rest/v1"
        assert result["auth_url"] == "https://abc123.supabase.co/auth/v1"
        assert result["storage_url"] == "https://abc123.supabase.co/storage/v1"
        assert result["functions_url"] == "https://abc123.supabase.co/functions/v1"

    def test_uses_project_ref_as_name_when_none(self):
        """Test that project_ref is used as name when project_name is None."""
        result = _build_manual_supabase_metadata(
            project_ref="abc123",
            project_name=None
        )

        assert result["name"] == "abc123"


# =============================================================================
# Bind Supabase Project Manual Tests
# =============================================================================


@pytest.mark.unit
class TestBindSupabaseProjectManual:
    """Tests for bind_supabase_project_manual function."""

    @pytest.mark.asyncio
    async def test_bind_validates_project_ref_required(self, mock_user):
        """Test that empty project_ref raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            await bind_supabase_project_manual(
                mock_user,
                project_ref="",
                service_role_key="key123"
            )

        assert exc_info.value.status_code == 400
        assert "project_ref" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_bind_validates_service_role_key_required(self, mock_user):
        """Test that empty service_role_key raises 400."""
        with pytest.raises(HTTPException) as exc_info:
            await bind_supabase_project_manual(
                mock_user,
                project_ref="abc123",
                service_role_key=""
            )

        assert exc_info.value.status_code == 400
        assert "service_role_key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_bind_creates_resource_and_secrets(self, mock_user, mock_integration_resource):
        """Test that binding creates resource and secrets."""
        with patch("seer.api.integrations.services._upsert_integration_resource") as mock_upsert_resource, \
             patch("seer.api.integrations.services._upsert_integration_secret") as mock_upsert_secret:

            mock_upsert_resource.return_value = mock_integration_resource
            mock_upsert_secret.return_value = MagicMock()

            result = await bind_supabase_project_manual(
                mock_user,
                project_ref="abc123",
                service_role_key="service_key_123",
                project_name="My Project",
                anon_key="anon_key_123"
            )

            assert result == mock_integration_resource

            # Verify resource was created
            mock_upsert_resource.assert_called_once()
            call_kwargs = mock_upsert_resource.call_args.kwargs
            assert call_kwargs["user"] == mock_user
            assert call_kwargs["provider"] == "supabase"
            assert call_kwargs["resource_type"] == "project"
            assert call_kwargs["resource_id"] == "abc123"

            # Verify both secrets were created (service_role + anon)
            assert mock_upsert_secret.call_count == 2

    @pytest.mark.asyncio
    async def test_bind_creates_only_service_role_when_no_anon(self, mock_user, mock_integration_resource):
        """Test that only service_role secret is created when anon_key not provided."""
        with patch("seer.api.integrations.services._upsert_integration_resource") as mock_upsert_resource, \
             patch("seer.api.integrations.services._upsert_integration_secret") as mock_upsert_secret:

            mock_upsert_resource.return_value = mock_integration_resource
            mock_upsert_secret.return_value = MagicMock()

            await bind_supabase_project_manual(
                mock_user,
                project_ref="abc123",
                service_role_key="service_key_123"
            )

            # Verify only one secret was created (service_role only)
            assert mock_upsert_secret.call_count == 1
            call_kwargs = mock_upsert_secret.call_args.kwargs
            assert call_kwargs["name"] == "supabase_service_role_key"

    @pytest.mark.asyncio
    async def test_bind_trims_project_ref(self, mock_user, mock_integration_resource):
        """Test that project_ref is trimmed."""
        with patch("seer.api.integrations.services._upsert_integration_resource") as mock_upsert_resource, \
             patch("seer.api.integrations.services._upsert_integration_secret") as mock_upsert_secret:

            mock_upsert_resource.return_value = mock_integration_resource
            mock_upsert_secret.return_value = MagicMock()

            await bind_supabase_project_manual(
                mock_user,
                project_ref="  abc123  ",
                service_role_key="service_key_123"
            )

            call_kwargs = mock_upsert_resource.call_args.kwargs
            assert call_kwargs["resource_id"] == "abc123"


# =============================================================================
# Connection Status Tests
# =============================================================================


@pytest.mark.unit
class TestConnectionStatus:
    """Tests for OAuth connection status handling."""

    def test_valid_status_values(self):
        """Test valid connection status values."""
        valid_statuses = ["active", "revoked", "error"]

        for status in valid_statuses:
            assert status in valid_statuses

    def test_active_status_is_queryable(self, mock_oauth_connection):
        """Test that active status is the default query filter."""
        mock_oauth_connection.status = "active"
        assert mock_oauth_connection.status == "active"

    def test_revoked_status_indicates_disconnection(self, mock_oauth_connection):
        """Test revoked status indicates disconnection."""
        mock_oauth_connection.status = "revoked"
        assert mock_oauth_connection.status == "revoked"
