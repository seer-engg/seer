"""
Unit tests for integration metadata API endpoint.
"""
import pytest

from seer.api.integrations.metadata_models import (
    IntegrationMetadata,
    IntegrationIcon,
    IntegrationScope,
)
from seer.services.integrations.metadata import (
    get_all_integration_metadata,
    get_integration_config,
    get_display_name,
    get_oauth_provider,
    INTEGRATION_CONFIGS,
)


@pytest.mark.unit
class TestIntegrationMetadataModels:
    """Test Pydantic models for integration metadata."""

    def test_integration_icon_model(self):
        """Test IntegrationIcon model validation."""
        icon = IntegrationIcon(type="url", value="https://example.com/icon.svg")
        assert icon.type == "url"
        assert icon.value == "https://example.com/icon.svg"

    def test_integration_scope_model(self):
        """Test IntegrationScope model validation."""
        scope = IntegrationScope(
            value="https://www.googleapis.com/auth/gmail.readonly",
            display_name="Read emails",
            description="Read email messages"
        )
        assert scope.value == "https://www.googleapis.com/auth/gmail.readonly"
        assert scope.display_name == "Read emails"

    def test_integration_metadata_model(self):
        """Test IntegrationMetadata model validation."""
        metadata = IntegrationMetadata(
            type="gmail",
            display_name="Gmail",
            oauth_provider="google",
            requires_oauth=True,
            icon=IntegrationIcon(type="url", value="https://example.com/gmail.svg"),
            default_scopes=["https://www.googleapis.com/auth/gmail.readonly"],
            scopes=[],
        )
        assert metadata.type == "gmail"
        assert metadata.display_name == "Gmail"
        assert metadata.oauth_provider == "google"


@pytest.mark.unit
class TestIntegrationMetadataService:
    """Test integration metadata service functions."""

    def test_get_all_integration_metadata_returns_dict(self):
        """Test that get_all_integration_metadata returns expected structure."""
        result = get_all_integration_metadata()

        assert "integrations" in result
        assert "provider_to_types" in result
        assert isinstance(result["integrations"], list)
        assert isinstance(result["provider_to_types"], dict)

    def test_get_all_integration_metadata_has_required_integrations(self):
        """Test that all configured integrations are returned."""
        result = get_all_integration_metadata()
        types = {i["type"] for i in result["integrations"]}

        # Check that key integrations are present
        assert "gmail" in types
        assert "google_drive" in types
        assert "google_sheets" in types
        assert "linkedin" in types
        assert "discord" in types
        assert "supabase" in types

    def test_integration_metadata_has_required_fields(self):
        """Test that each integration has all required fields."""
        result = get_all_integration_metadata()

        for integration in result["integrations"]:
            assert "type" in integration
            assert "display_name" in integration
            assert "oauth_provider" in integration
            assert "requires_oauth" in integration
            assert "icon" in integration
            assert "default_scopes" in integration
            assert "scopes" in integration

    def test_provider_to_types_mapping(self):
        """Test provider_to_types mapping is correct."""
        result = get_all_integration_metadata()
        provider_map = result["provider_to_types"]

        # Google provider should have multiple integration types
        assert "google" in provider_map
        assert "gmail" in provider_map["google"]
        assert "google_drive" in provider_map["google"]
        assert "google_sheets" in provider_map["google"]

        # LinkedIn should have its own provider
        assert "linkedin" in provider_map
        assert "linkedin" in provider_map["linkedin"]

    def test_get_integration_config(self):
        """Test get_integration_config returns correct config."""
        config = get_integration_config("gmail")
        assert config is not None
        assert config["display_name"] == "Gmail"
        assert config["oauth_provider"] == "google"

        # Unknown integration returns None
        assert get_integration_config("unknown") is None

    def test_get_display_name(self):
        """Test get_display_name returns correct names."""
        assert get_display_name("gmail") == "Gmail"
        assert get_display_name("google_drive") == "Google Drive"
        assert get_display_name("linkedin") == "LinkedIn"

        # Unknown integration returns formatted type
        assert get_display_name("unknown_type") == "Unknown Type"

    def test_get_oauth_provider_service(self):
        """Test get_oauth_provider from metadata service."""
        assert get_oauth_provider("gmail") == "google"
        assert get_oauth_provider("google_drive") == "google"
        assert get_oauth_provider("linkedin") == "linkedin"

        # Unknown returns None
        assert get_oauth_provider("unknown") is None


@pytest.mark.unit
class TestIntegrationConfigs:
    """Test static integration configurations."""

    def test_all_configs_have_required_keys(self):
        """Test that all configs have required keys."""
        required_keys = {"display_name", "oauth_provider", "requires_oauth", "icon"}

        for int_type, config in INTEGRATION_CONFIGS.items():
            for key in required_keys:
                assert key in config, f"Config for {int_type} missing key: {key}"

    def test_all_icons_have_type_and_value(self):
        """Test that all icon configs have type and value."""
        for int_type, config in INTEGRATION_CONFIGS.items():
            icon = config["icon"]
            assert "type" in icon, f"Icon for {int_type} missing type"
            assert "value" in icon, f"Icon for {int_type} missing value"
            assert icon["type"] in {"url", "lucide", "svg"}, f"Invalid icon type for {int_type}"

    def test_all_scopes_have_required_fields(self):
        """Test that all scope configs have required fields."""
        for int_type, config in INTEGRATION_CONFIGS.items():
            for scope in config.get("scopes", []):
                assert "value" in scope, f"Scope in {int_type} missing value"
                assert "display_name" in scope, f"Scope in {int_type} missing display_name"
