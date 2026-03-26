"""
Unit tests for integrations services pure logic.

Tests pure functions: serialization, fingerprinting, provider mapping.
Heavy mock tests for DB operations have been moved to E2E tests.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import hashlib

import pytest
from fastapi import HTTPException

from seer.api.integrations.services import (
    serialize_integration_resource,
    serialize_integration_secret,
    bind_supabase_project_manual,
    _fingerprint_secret,
    _format_supabase_secret_name,
    _build_manual_supabase_metadata,
)
from seer.services.integrations.auth.oauth import get_oauth_provider


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


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
# Serialization Tests
# =============================================================================


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
# Bind Supabase Project Validation Tests
# =============================================================================


class TestBindSupabaseProjectManualValidation:
    """Tests for bind_supabase_project_manual validation logic."""

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
