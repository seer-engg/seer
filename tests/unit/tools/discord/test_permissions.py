"""Unit tests for Discord permission calculations and helpers."""

import pytest

from seer.tools.discord.permissions import (
    DiscordPermission,
    calculate_permissions,
    get_permission_names,
    has_permission,
    TOOL_PERMISSIONS_MAP,
)


class TestDiscordPermission:
    """Test DiscordPermission IntFlag enum."""

    def test_permission_values(self):
        """Test that permission constants have correct bitfield values."""
        assert DiscordPermission.VIEW_CHANNEL == 1024
        assert DiscordPermission.SEND_MESSAGES == 2048
        assert DiscordPermission.MANAGE_CHANNELS == 16
        assert DiscordPermission.EMBED_LINKS == 16384
        assert DiscordPermission.ATTACH_FILES == 32768
        assert DiscordPermission.READ_MESSAGE_HISTORY == 65536
        assert DiscordPermission.ADD_REACTIONS == 64
        assert DiscordPermission.MANAGE_MESSAGES == 8192

    def test_permission_bitwise_or(self):
        """Test combining permissions with bitwise OR."""
        combined = DiscordPermission.VIEW_CHANNEL | DiscordPermission.SEND_MESSAGES
        assert combined == 3072  # 1024 + 2048

        combined_three = (
            DiscordPermission.VIEW_CHANNEL
            | DiscordPermission.SEND_MESSAGES
            | DiscordPermission.ADD_REACTIONS
        )
        assert combined_three == 3136  # 1024 + 2048 + 64


class TestCalculatePermissions:
    """Test calculate_permissions function."""

    def test_single_tool_send_channel_message(self):
        """Test permissions for discord_send_channel_message."""
        perms = calculate_permissions(["discord_send_channel_message"])
        assert perms == 3072  # VIEW_CHANNEL | SEND_MESSAGES

    def test_single_tool_send_direct_message(self):
        """Test permissions for discord_send_direct_message."""
        perms = calculate_permissions(["discord_send_direct_message"])
        assert perms == 2048  # SEND_MESSAGES only

    def test_single_tool_find_user(self):
        """Test permissions for discord_find_user."""
        perms = calculate_permissions(["discord_find_user"])
        assert perms == 1024  # VIEW_CHANNEL only

    def test_multiple_tools_overlapping_permissions(self):
        """Test that overlapping permissions are merged correctly."""
        # Both tools need VIEW_CHANNEL, so result should still be 3072
        perms = calculate_permissions([
            "discord_send_channel_message",  # VIEW_CHANNEL | SEND_MESSAGES
            "discord_find_user",             # VIEW_CHANNEL
        ])
        assert perms == 3072

    def test_multiple_tools_different_permissions(self):
        """Test merging different permissions."""
        perms = calculate_permissions([
            "discord_send_direct_message",   # SEND_MESSAGES (2048)
            "discord_find_user",             # VIEW_CHANNEL (1024)
        ])
        assert perms == 3072  # VIEW_CHANNEL | SEND_MESSAGES

    def test_unknown_tool_ignored(self):
        """Test that unknown tool names are silently ignored."""
        perms = calculate_permissions(["unknown_tool"])
        assert perms == 0

    def test_empty_tool_list(self):
        """Test with empty tool list."""
        perms = calculate_permissions([])
        assert perms == 0

    def test_mixed_known_and_unknown_tools(self):
        """Test with mix of known and unknown tools."""
        perms = calculate_permissions([
            "discord_send_channel_message",
            "unknown_tool",
            "discord_find_user",
        ])
        assert perms == 3072


class TestGetPermissionNames:
    """Test get_permission_names function."""

    def test_single_permission(self):
        """Test converting single permission to name."""
        names = get_permission_names(1024)
        assert names == {"VIEW_CHANNEL"}

    def test_multiple_permissions(self):
        """Test converting multiple permissions to names."""
        names = get_permission_names(3072)  # VIEW_CHANNEL | SEND_MESSAGES
        assert names == {"VIEW_CHANNEL", "SEND_MESSAGES"}

    def test_complex_permissions(self):
        """Test converting complex permission set."""
        # VIEW_CHANNEL | SEND_MESSAGES | ADD_REACTIONS
        names = get_permission_names(3136)
        assert names == {"VIEW_CHANNEL", "SEND_MESSAGES", "ADD_REACTIONS"}

    def test_zero_permissions(self):
        """Test with no permissions."""
        names = get_permission_names(0)
        assert names == set()

    def test_all_basic_permissions(self):
        """Test with all basic permissions combined."""
        all_perms = (
            DiscordPermission.VIEW_CHANNEL
            | DiscordPermission.SEND_MESSAGES
            | DiscordPermission.MANAGE_CHANNELS
            | DiscordPermission.EMBED_LINKS
            | DiscordPermission.ATTACH_FILES
        )
        names = get_permission_names(all_perms)
        assert "VIEW_CHANNEL" in names
        assert "SEND_MESSAGES" in names
        assert "MANAGE_CHANNELS" in names
        assert "EMBED_LINKS" in names
        assert "ATTACH_FILES" in names


class TestHasPermission:
    """Test has_permission function."""

    def test_exact_match(self):
        """Test when granted exactly matches required."""
        assert has_permission(3072, 3072) is True

    def test_granted_includes_required(self):
        """Test when granted includes more than required."""
        granted = 3136  # VIEW_CHANNEL | SEND_MESSAGES | ADD_REACTIONS
        required = 3072  # VIEW_CHANNEL | SEND_MESSAGES
        assert has_permission(granted, required) is True

    def test_granted_missing_some_required(self):
        """Test when granted is missing some required permissions."""
        granted = 1024  # VIEW_CHANNEL only
        required = 3072  # VIEW_CHANNEL | SEND_MESSAGES
        assert has_permission(granted, required) is False

    def test_granted_has_different_permissions(self):
        """Test when granted has different permissions than required."""
        granted = 64  # ADD_REACTIONS
        required = 2048  # SEND_MESSAGES
        assert has_permission(granted, required) is False

    def test_zero_required_permissions(self):
        """Test with no required permissions (should always pass)."""
        assert has_permission(3072, 0) is True
        assert has_permission(0, 0) is True

    def test_zero_granted_permissions(self):
        """Test with no granted permissions but something required."""
        assert has_permission(0, 2048) is False

    def test_subset_permission_check(self):
        """Test checking for subset of permissions."""
        granted = 3072  # VIEW_CHANNEL | SEND_MESSAGES
        assert has_permission(granted, 1024) is True  # Has VIEW_CHANNEL
        assert has_permission(granted, 2048) is True  # Has SEND_MESSAGES
        assert has_permission(granted, 64) is False   # Doesn't have ADD_REACTIONS


class TestToolPermissionsMap:
    """Test TOOL_PERMISSIONS_MAP configuration."""

    def test_map_contains_expected_tools(self):
        """Test that map contains all expected Discord tools."""
        expected_tools = {
            "discord_send_channel_message",
            "discord_send_direct_message",
            "discord_find_user",
        }
        assert set(TOOL_PERMISSIONS_MAP.keys()) == expected_tools

    def test_send_channel_message_permissions(self):
        """Test permissions for send_channel_message."""
        perms = TOOL_PERMISSIONS_MAP["discord_send_channel_message"]
        assert perms == 3072  # VIEW_CHANNEL | SEND_MESSAGES

    def test_send_direct_message_permissions(self):
        """Test permissions for send_direct_message."""
        perms = TOOL_PERMISSIONS_MAP["discord_send_direct_message"]
        assert perms == 2048  # SEND_MESSAGES

    def test_find_user_permissions(self):
        """Test permissions for find_user."""
        perms = TOOL_PERMISSIONS_MAP["discord_find_user"]
        assert perms == 1024  # VIEW_CHANNEL


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_large_permission_bitfield(self):
        """Test with large permission values."""
        # MANAGE_ROLES = 268435456
        perms = calculate_permissions([])
        assert perms == 0

        # Test has_permission with large values
        granted = DiscordPermission.MANAGE_ROLES
        required = DiscordPermission.MANAGE_ROLES
        assert has_permission(granted, required) is True

    def test_permission_names_ordering_independent(self):
        """Test that permission names are returned as set (order-independent)."""
        names1 = get_permission_names(3072)
        names2 = get_permission_names(3072)
        assert names1 == names2
        assert isinstance(names1, set)

    def test_duplicate_tools_in_list(self):
        """Test that duplicate tools don't double permissions."""
        perms = calculate_permissions([
            "discord_send_channel_message",
            "discord_send_channel_message",  # Duplicate
        ])
        assert perms == 3072  # Should still be same as single occurrence
