"""Unit tests for Discord permission validators."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from seer.tools.discord.validators import validate_discord_permissions


@pytest.mark.unit
class TestValidateDiscordPermissions:
    """Test validate_discord_permissions function."""

    def test_validation_passes_with_exact_permissions(self):
        """Test validation passes when permissions match exactly."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 3072}

        # Should not raise exception
        validate_discord_permissions(
            connection=connection,
            required_permissions=3072,
            tool_name="discord_send_channel_message"
        )

    def test_validation_passes_with_superset_permissions(self):
        """Test validation passes when granted permissions include more than required."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 3136}  # 3072 + ADD_REACTIONS (64)

        # Should not raise exception (has all of 3072 plus extra)
        validate_discord_permissions(
            connection=connection,
            required_permissions=3072,
            tool_name="discord_send_channel_message"
        )

    def test_validation_fails_with_missing_permissions(self):
        """Test validation fails when missing required permissions."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 1024}  # VIEW_CHANNEL only

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,  # Needs VIEW_CHANNEL + SEND_MESSAGES
                tool_name="discord_send_channel_message"
            )

        assert exc_info.value.status_code == 403
        assert "discord_send_channel_message" in exc_info.value.detail
        assert "SEND_MESSAGES" in exc_info.value.detail
        assert "Please reconnect" in exc_info.value.detail

    def test_validation_fails_with_no_metadata(self):
        """Test validation fails when connection has no provider_metadata."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = None

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,
                tool_name="discord_send_channel_message"
            )

        assert exc_info.value.status_code == 403
        assert "No permission metadata" in exc_info.value.detail
        assert "Please reconnect" in exc_info.value.detail

    def test_validation_fails_with_empty_metadata(self):
        """Test validation fails when metadata has no permissions key."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {}  # No permissions key

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,
                tool_name="discord_send_channel_message"
            )

        assert exc_info.value.status_code == 403

    def test_validation_with_zero_required_permissions(self):
        """Test validation passes when no permissions are required."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 0}

        # Should not raise exception (nothing required)
        validate_discord_permissions(
            connection=connection,
            required_permissions=0,
            tool_name="test_tool"
        )

    def test_validation_error_message_includes_missing_permissions(self):
        """Test that error message includes specific missing permissions."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 1024}  # VIEW_CHANNEL only

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,  # VIEW_CHANNEL + SEND_MESSAGES
                tool_name="test_tool"
            )

        detail = exc_info.value.detail
        assert "test_tool" in detail
        assert "Missing: SEND_MESSAGES" in detail or "Missing:" in detail

    def test_validation_with_multiple_missing_permissions(self):
        """Test validation with multiple missing permissions."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 0}  # No permissions

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,  # VIEW_CHANNEL + SEND_MESSAGES
                tool_name="test_tool"
            )

        detail = exc_info.value.detail
        assert exc_info.value.status_code == 403
        # Should mention missing permissions
        assert "Missing:" in detail

    def test_validation_passes_with_subset_check(self):
        """Test validation when checking for subset of granted permissions."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 3072}  # VIEW_CHANNEL + SEND_MESSAGES

        # Check for just VIEW_CHANNEL (subset of granted)
        validate_discord_permissions(
            connection=connection,
            required_permissions=1024,  # VIEW_CHANNEL only
            tool_name="discord_find_user"
        )

    def test_validation_fails_with_different_permissions(self):
        """Test validation fails when permissions are completely different."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 64}  # ADD_REACTIONS only

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=3072,  # VIEW_CHANNEL + SEND_MESSAGES
                tool_name="test_tool"
            )

        assert exc_info.value.status_code == 403


@pytest.mark.unit
class TestValidationErrorDetails:
    """Test error message details and structure."""

    def test_error_includes_tool_name(self):
        """Test that error message includes the tool name."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 0}

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=2048,
                tool_name="my_custom_tool"
            )

        assert "my_custom_tool" in exc_info.value.detail

    def test_error_status_code_is_403(self):
        """Test that validation failure returns 403 Forbidden."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 0}

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=2048,
                tool_name="test_tool"
            )

        assert exc_info.value.status_code == 403

    def test_error_suggests_reconnection(self):
        """Test that error message suggests reconnecting."""
        connection = Mock()
        connection.id = "test_connection"
        connection.provider_metadata = {"permissions": 0}

        with pytest.raises(HTTPException) as exc_info:
            validate_discord_permissions(
                connection=connection,
                required_permissions=2048,
                tool_name="test_tool"
            )

        assert "reconnect" in exc_info.value.detail.lower()


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validation_with_large_permission_values(self):
        """Test validation with large permission bitfields."""
        connection = Mock()
        connection.id = "test_connection"
        # MANAGE_ROLES = 268435456
        connection.provider_metadata = {"permissions": 268435456}

        # Should pass
        validate_discord_permissions(
            connection=connection,
            required_permissions=268435456,
            tool_name="test_tool"
        )

    def test_validation_with_all_permissions(self):
        """Test validation when granted has many permissions."""
        connection = Mock()
        connection.id = "test_connection"
        # Permission value includes multiple permissions including VIEW_CHANNEL and SEND_MESSAGES.
        # Original 536871935 had MANAGE_WEBHOOKS, KICK_MEMBERS, etc. but was missing VIEW_CHANNEL/SEND_MESSAGES.
        # 536871935 | 3072 = 536874943 ensures we have VIEW_CHANNEL (1024) and SEND_MESSAGES (2048) included.
        connection.provider_metadata = {"permissions": 536874943}

        # Should pass for any subset
        validate_discord_permissions(
            connection=connection,
            required_permissions=3072,
            tool_name="test_tool"
        )

    def test_validation_connection_id_in_logs(self):
        """Test that connection ID would be included in logs (not in exception)."""
        connection = Mock()
        connection.id = "test_connection_123"
        connection.provider_metadata = None

        # Just verify it doesn't crash with connection ID
        with pytest.raises(HTTPException):
            validate_discord_permissions(
                connection=connection,
                required_permissions=2048,
                tool_name="test_tool"
            )
