"""Unit tests for DiscordProvider permission calculation and incremental auth."""

from unittest.mock import Mock

import pytest

from seer.services.integrations.providers.discord import DiscordProvider
from seer.services.integrations.providers.base import OAuthAuthorizeContext


@pytest.mark.unit
class TestDiscordProviderScopeHandling:
    """Test DiscordProvider.get_oauth_scope()."""

    def test_get_oauth_scope_always_returns_bot(self):
        """Test that get_oauth_scope always returns 'bot' for Discord."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]

        scope = provider.get_oauth_scope(context)
        assert scope == "bot"

    def test_get_oauth_scope_ignores_context(self):
        """Test that get_oauth_scope ignores context content."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = []

        scope = provider.get_oauth_scope(context)
        assert scope == "bot"


@pytest.mark.unit
class TestCalculateRequestedPermissions:
    """Test DiscordProvider._calculate_requested_permissions()."""

    def test_single_tool_calculation(self):
        """Test permission calculation with single tool."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]

        perms = provider._calculate_requested_permissions(context)
        assert perms == 3072  # VIEW_CHANNEL | SEND_MESSAGES

    def test_multiple_tools_calculation(self):
        """Test permission calculation with multiple tools."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = [
            "discord_send_channel_message",
            "discord_find_user",
        ]

        perms = provider._calculate_requested_permissions(context)
        assert perms == 3072  # Combined permissions (both need VIEW_CHANNEL)

    def test_empty_scopes_returns_default(self):
        """Test that empty scopes returns DEFAULT_PERMISSIONS."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = []

        perms = provider._calculate_requested_permissions(context)
        assert perms == 3072  # DEFAULT_PERMISSIONS

    def test_none_scopes_returns_default(self):
        """Test that None scopes returns DEFAULT_PERMISSIONS."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = None

        perms = provider._calculate_requested_permissions(context)
        assert perms == 3072

    def test_unknown_tools_returns_default(self):
        """Test that unknown tool names return DEFAULT_PERMISSIONS."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["unknown_tool", "another_unknown"]

        perms = provider._calculate_requested_permissions(context)
        assert perms == 3072  # Falls back to default


@pytest.mark.unit
class TestBuildAuthorizeKwargsFirstTime:
    """Test build_authorize_kwargs for first-time authorization."""

    def test_first_time_auth_no_existing_connection(self):
        """Test first-time auth with no existing connection."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]
        context.existing_connection = None

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        assert kwargs["state"] == "test_state"
        assert kwargs["scope"] == "bot"
        assert kwargs["permissions"] == 3072  # VIEW_CHANNEL | SEND_MESSAGES

    def test_first_time_auth_with_multiple_tools(self):
        """Test first-time auth with multiple tools."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = [
            "discord_send_direct_message",  # 2048
            "discord_find_user",            # 1024
        ]
        context.existing_connection = None

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        assert kwargs["permissions"] == 3072  # Combined: 1024 | 2048


@pytest.mark.unit
class TestBuildAuthorizeKwargsIncremental:
    """Test build_authorize_kwargs for incremental authorization."""

    def test_incremental_auth_new_permissions_needed(self):
        """Test incremental auth when new permissions are needed."""
        provider = DiscordProvider()

        # Mock existing connection with some permissions
        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 1024}  # VIEW_CHANNEL only

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]  # Needs 3072
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        # Should request merged permissions: 1024 | 3072 = 3072
        assert kwargs["permissions"] == 3072

    def test_incremental_auth_already_have_all_permissions(self):
        """Test incremental auth when already have all required permissions."""
        provider = DiscordProvider()

        # Mock existing connection with all required permissions
        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 3072}  # Already have all

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]  # Needs 3072
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        # Should keep existing permissions
        assert kwargs["permissions"] == 3072

    def test_incremental_auth_adding_new_permission(self):
        """Test incremental auth when adding a new distinct permission."""
        provider = DiscordProvider()

        # Mock existing connection with SEND_MESSAGES only
        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 2048}  # SEND_MESSAGES

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_find_user"]  # Needs VIEW_CHANNEL (1024)
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        # Should merge: 2048 | 1024 = 3072
        assert kwargs["permissions"] == 3072

    def test_incremental_auth_no_metadata(self):
        """Test incremental auth with existing connection but no metadata."""
        provider = DiscordProvider()

        # Mock existing connection without provider_metadata
        existing_conn = Mock()
        existing_conn.provider_metadata = None

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        # Should treat as first-time auth
        assert kwargs["permissions"] == 3072

    def test_incremental_auth_empty_metadata(self):
        """Test incremental auth with empty metadata dict."""
        provider = DiscordProvider()

        # Mock existing connection with empty metadata
        existing_conn = Mock()
        existing_conn.provider_metadata = {}  # No permissions key

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="test_state",
            scope="bot"
        )

        # Should use requested permissions (0 | 3072 = 3072)
        assert kwargs["permissions"] == 3072


@pytest.mark.unit
class TestBuildAuthorizeKwargsStateAndScope:
    """Test that state and scope are always included in kwargs."""

    def test_state_and_scope_always_included(self):
        """Test that state and scope parameters are always returned."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]
        context.existing_connection = None

        kwargs = provider.build_authorize_kwargs(
            context,
            state="custom_state_123",
            scope="bot"
        )

        assert "state" in kwargs
        assert "scope" in kwargs
        assert "permissions" in kwargs
        assert kwargs["state"] == "custom_state_123"
        assert kwargs["scope"] == "bot"

    def test_kwargs_structure(self):
        """Test that returned kwargs has expected structure."""
        provider = DiscordProvider()
        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = []
        context.existing_connection = None

        kwargs = provider.build_authorize_kwargs(
            context,
            state="state",
            scope="bot"
        )

        assert isinstance(kwargs, dict)
        assert len(kwargs) == 3  # state, scope, permissions
        assert all(isinstance(k, str) for k in kwargs.keys())


@pytest.mark.unit
class TestProviderAttributes:
    """Test DiscordProvider class attributes."""

    def test_provider_name(self):
        """Test provider attribute is 'discord'."""
        provider = DiscordProvider()
        assert provider.provider == "discord"

    def test_resource_types(self):
        """Test resource_types includes 'guild'."""
        provider = DiscordProvider()
        assert "guild" in provider.resource_types

    def test_default_permissions_value(self):
        """Test DEFAULT_PERMISSIONS is set correctly."""
        provider = DiscordProvider()
        assert provider.DEFAULT_PERMISSIONS == 3072  # VIEW_CHANNEL | SEND_MESSAGES


@pytest.mark.unit
class TestPermissionMerging:
    """Test permission merging logic in various scenarios."""

    def test_merge_non_overlapping_permissions(self):
        """Test merging completely different permissions."""
        provider = DiscordProvider()

        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 16}  # MANAGE_CHANNELS

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]  # 3072
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="state",
            scope="bot"
        )

        # Should merge: 16 | 3072 = 3088
        assert kwargs["permissions"] == 3088

    def test_merge_overlapping_permissions(self):
        """Test merging when permissions overlap."""
        provider = DiscordProvider()

        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 1024}  # VIEW_CHANNEL

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]  # 3072 (includes VIEW_CHANNEL)
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="state",
            scope="bot"
        )

        # Should merge: 1024 | 3072 = 3072 (VIEW_CHANNEL is already in 3072)
        assert kwargs["permissions"] == 3072

    def test_merge_zero_existing_permissions(self):
        """Test merging when existing permissions is 0."""
        provider = DiscordProvider()

        existing_conn = Mock()
        existing_conn.provider_metadata = {"permissions": 0}

        context = Mock(spec=OAuthAuthorizeContext)
        context.requested_scopes = ["discord_send_channel_message"]
        context.existing_connection = existing_conn

        kwargs = provider.build_authorize_kwargs(
            context,
            state="state",
            scope="bot"
        )

        assert kwargs["permissions"] == 3072
