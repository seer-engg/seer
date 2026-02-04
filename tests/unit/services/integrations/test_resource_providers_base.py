"""
Unit tests for services.integrations.resource_providers.base module.

Tests ResourceProvider and ResourceProviderRegistry classes.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import MagicMock

import pytest


# =============================================================================
# ResourceProvider Tests
# =============================================================================


@pytest.mark.unit
class TestResourceProvider:
    """Tests for ResourceProvider base class."""

    @pytest.fixture
    def sample_provider(self):
        """Create a sample ResourceProvider subclass."""
        from seer.services.integrations.resource_providers.base import ResourceProvider

        class GoogleProvider(ResourceProvider):
            provider = "google"
            aliases = {"googledrive", "gmail", "googlesheets"}
            resource_configs = {
                "google_drive_file": {"display_field": "name", "value_field": "id"},
                "google_sheet": {"display_field": "name", "value_field": "id"},
            }

        return GoogleProvider()

    def test_matches_provider_exact_match(self, sample_provider):
        """Test matches_provider with exact provider name."""
        assert sample_provider.matches_provider("google") is True

    def test_matches_provider_alias_match(self, sample_provider):
        """Test matches_provider with alias name."""
        assert sample_provider.matches_provider("googledrive") is True
        assert sample_provider.matches_provider("gmail") is True
        assert sample_provider.matches_provider("googlesheets") is True

    def test_matches_provider_no_match(self, sample_provider):
        """Test matches_provider returns False for non-matching provider."""
        assert sample_provider.matches_provider("github") is False
        assert sample_provider.matches_provider("Google") is False  # Case-sensitive

    def test_supports_resource_type_supported(self, sample_provider):
        """Test supports_resource_type returns True for supported types."""
        assert sample_provider.supports_resource_type("google_drive_file") is True
        assert sample_provider.supports_resource_type("google_sheet") is True

    def test_supports_resource_type_unsupported(self, sample_provider):
        """Test supports_resource_type returns False for unsupported types."""
        assert sample_provider.supports_resource_type("github_repo") is False
        assert sample_provider.supports_resource_type("unknown_type") is False

    def test_get_supported_resource_types(self, sample_provider):
        """Test get_supported_resource_types returns all types."""
        types = sample_provider.get_supported_resource_types()
        assert "google_drive_file" in types
        assert "google_sheet" in types
        assert len(types) == 2

    def test_get_resource_config_found(self, sample_provider):
        """Test get_resource_config returns config for known type."""
        config = sample_provider.get_resource_config("google_drive_file")
        assert config == {"display_field": "name", "value_field": "id"}

    def test_get_resource_config_not_found(self, sample_provider):
        """Test get_resource_config returns None for unknown type."""
        config = sample_provider.get_resource_config("unknown_type")
        assert config is None


@pytest.mark.unit
class TestResourceProviderListResources:
    """Tests for ResourceProvider.list_resources method."""

    @pytest.mark.asyncio
    async def test_list_resources_raises_not_supported(self):
        """Test base list_resources raises HTTPException."""
        from fastapi import HTTPException
        from seer.services.integrations.resource_providers.base import ResourceProvider

        class MinimalProvider(ResourceProvider):
            provider = "minimal"
            aliases = set()
            resource_configs = {}

        provider = MinimalProvider()

        with pytest.raises(HTTPException) as exc_info:
            await provider.list_resources(
                resource_type="any_type",
                query=None,
                parent_id=None,
                page_token=None,
                page_size=10,
                filter_params=None,
                depends_on_values=None,
            )

        assert exc_info.value.status_code == 400
        assert "does not support" in exc_info.value.detail


# =============================================================================
# ResourceProviderRegistry Tests
# =============================================================================


@pytest.mark.unit
class TestResourceProviderRegistry:
    """Tests for ResourceProviderRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry instance."""
        from seer.services.integrations.resource_providers.base import ResourceProviderRegistry
        return ResourceProviderRegistry()

    @pytest.fixture
    def mock_google_provider(self):
        """Create a mock Google provider."""
        provider = MagicMock()
        provider.provider = "google"
        provider.aliases = {"googledrive", "gmail"}
        provider.resource_configs = {"google_drive_file": {"key": "value"}}
        provider.supports_resource_type = MagicMock(side_effect=lambda t: t == "google_drive_file")
        provider.get_resource_config = MagicMock(side_effect=lambda t: {"key": "value"} if t == "google_drive_file" else None)
        return provider

    @pytest.fixture
    def mock_github_provider(self):
        """Create a mock GitHub provider."""
        provider = MagicMock()
        provider.provider = "github"
        provider.aliases = set()
        provider.resource_configs = {"github_repo": {"key": "value"}}
        provider.supports_resource_type = MagicMock(side_effect=lambda t: t == "github_repo")
        provider.get_resource_config = MagicMock(side_effect=lambda t: {"key": "value"} if t == "github_repo" else None)
        return provider

    def test_register_adds_by_provider_name(self, registry, mock_google_provider):
        """Test register adds provider by main provider name."""
        registry.register(mock_google_provider)

        result = registry.get("google")
        assert result == mock_google_provider

    def test_register_adds_by_aliases(self, registry, mock_google_provider):
        """Test register adds provider by all aliases."""
        registry.register(mock_google_provider)

        assert registry.get("googledrive") == mock_google_provider
        assert registry.get("gmail") == mock_google_provider

    def test_get_found(self, registry, mock_google_provider):
        """Test get returns provider when found."""
        registry.register(mock_google_provider)

        result = registry.get("google")
        assert result == mock_google_provider

    def test_get_not_found(self, registry):
        """Test get returns None when provider not found."""
        result = registry.get("nonexistent")
        assert result is None

    def test_find_by_resource_type_found(self, registry, mock_google_provider, mock_github_provider):
        """Test find_by_resource_type finds provider supporting the type."""
        registry.register(mock_google_provider)
        registry.register(mock_github_provider)

        result = registry.find_by_resource_type("google_drive_file")
        assert result == mock_google_provider

        result = registry.find_by_resource_type("github_repo")
        assert result == mock_github_provider

    def test_find_by_resource_type_not_found(self, registry, mock_google_provider):
        """Test find_by_resource_type returns None when type not supported."""
        registry.register(mock_google_provider)

        result = registry.find_by_resource_type("unknown_type")
        assert result is None

    def test_supports_true(self, registry, mock_google_provider):
        """Test supports returns True when provider supports type."""
        registry.register(mock_google_provider)

        result = registry.supports("google", "google_drive_file")
        assert result is True

    def test_supports_false_provider_not_found(self, registry):
        """Test supports returns False when provider not found."""
        result = registry.supports("nonexistent", "any_type")
        assert result is False

    def test_supports_false_type_not_supported(self, registry, mock_google_provider):
        """Test supports returns False when type not supported."""
        registry.register(mock_google_provider)

        result = registry.supports("google", "unsupported_type")
        assert result is False

    def test_get_config_found(self, registry, mock_google_provider):
        """Test get_config returns config when found."""
        registry.register(mock_google_provider)

        result = registry.get_config("google", "google_drive_file")
        assert result == {"key": "value"}

    def test_get_config_provider_not_found(self, registry):
        """Test get_config returns None when provider not found."""
        result = registry.get_config("nonexistent", "any_type")
        assert result is None

    def test_get_config_type_not_found(self, registry, mock_google_provider):
        """Test get_config returns None when type not found."""
        registry.register(mock_google_provider)

        result = registry.get_config("google", "unknown_type")
        assert result is None

    def test_resource_configs_merged(self, registry, mock_google_provider, mock_github_provider):
        """Test resource_configs merges all provider configs."""
        registry.register(mock_google_provider)
        registry.register(mock_github_provider)

        result = registry.resource_configs()

        assert "google_drive_file" in result
        assert "github_repo" in result

    def test_resource_configs_no_duplicates(self, registry, mock_google_provider):
        """Test resource_configs doesn't duplicate from aliases."""
        registry.register(mock_google_provider)

        # Access via alias to ensure it's registered
        assert registry.get("googledrive") is not None

        result = registry.resource_configs()

        # Should only have configs once, not duplicated for each alias
        assert len(result) == 1


# =============================================================================
# ResourceContext Tests
# =============================================================================


@pytest.mark.unit
class TestResourceContext:
    """Tests for ResourceContext dataclass."""

    def test_resource_context_with_oauth_token(self):
        """Test creating ResourceContext with OAuth token."""
        from seer.services.integrations.resource_providers.base import ResourceContext

        mock_user = MagicMock()
        context = ResourceContext(user=mock_user, access_token="oauth_token_123")

        assert context.user == mock_user
        assert context.access_token == "oauth_token_123"
        assert context.bot_token is None
        assert context.api_key is None

    def test_resource_context_with_bot_token(self):
        """Test creating ResourceContext with bot token."""
        from seer.services.integrations.resource_providers.base import ResourceContext

        mock_user = MagicMock()
        context = ResourceContext(user=mock_user, bot_token="bot_token_abc")

        assert context.bot_token == "bot_token_abc"
        assert context.access_token is None

    def test_resource_context_with_custom_auth(self):
        """Test creating ResourceContext with custom auth."""
        from seer.services.integrations.resource_providers.base import ResourceContext

        mock_user = MagicMock()
        custom = {"header": "X-Custom-Auth", "value": "secret"}
        context = ResourceContext(user=mock_user, custom_auth=custom)

        assert context.custom_auth == custom

    def test_resource_context_immutable(self):
        """Test ResourceContext is immutable (frozen dataclass)."""
        from seer.services.integrations.resource_providers.base import ResourceContext

        mock_user = MagicMock()
        context = ResourceContext(user=mock_user, access_token="token")

        with pytest.raises(Exception):  # FrozenInstanceError
            context.access_token = "new_token"
