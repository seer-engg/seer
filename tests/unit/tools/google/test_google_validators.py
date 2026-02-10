"""Unit tests for Google scope validators."""

from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from seer.tools.google.validators import validate_google_scopes


@pytest.mark.unit
class TestValidateGoogleScopes:
    """Test validate_google_scopes function."""

    def test_validation_passes_with_exact_scopes(self):
        """Test validation passes when scopes match exactly."""
        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.send"

        # Should not raise exception
        validate_google_scopes(
            connection=connection,
            required_scopes=["https://www.googleapis.com/auth/gmail.send"],
            tool_name="gmail_send_email"
        )

    def test_validation_passes_with_multiple_scopes(self):
        """Test validation passes when connection has all required scopes."""
        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/gmail.readonly"

        # Should not raise exception
        validate_google_scopes(
            connection=connection,
            required_scopes=[
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.readonly"
            ],
            tool_name="gmail_tool"
        )

    def test_validation_passes_with_superset_scopes(self):
        """Test validation passes when granted has more scopes than required."""
        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/drive"

        # Should not raise exception (has gmail.send plus extra)
        validate_google_scopes(
            connection=connection,
            required_scopes=["https://www.googleapis.com/auth/gmail.send"],
            tool_name="gmail_send_email"
        )

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_validation_fails_with_missing_scopes(self, mock_has_required):
        """Test validation fails when missing required scopes."""
        mock_has_required.side_effect = lambda granted, required: False

        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.readonly"

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="gmail_send_email"
            )

        assert exc_info.value.status_code == 403
        assert "gmail_send_email" in exc_info.value.detail
        assert "Missing required Google OAuth scopes" in exc_info.value.detail
        assert "Please reconnect" in exc_info.value.detail

    def test_validation_with_empty_scopes(self):
        """Test validation with empty granted scopes."""
        connection = Mock()
        connection.scopes = ""

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="gmail_send_email"
            )

        assert exc_info.value.status_code == 403

    def test_validation_with_none_scopes(self):
        """Test validation with None granted scopes."""
        connection = Mock()
        connection.scopes = None

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="gmail_send_email"
            )

        assert exc_info.value.status_code == 403

    def test_validation_with_empty_required_scopes(self):
        """Test validation passes when no scopes are required."""
        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.send"

        # Should not raise exception (nothing required)
        validate_google_scopes(
            connection=connection,
            required_scopes=[],
            tool_name="test_tool"
        )

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_validation_error_message_includes_missing_scopes(self, mock_has_required):
        """Test that error message includes specific missing scopes."""
        # Mock to return False for the overall check, True for individual scopes except one
        def mock_check(granted, required):
            if len(required) == 1:
                # Individual scope check
                return required[0] != "https://www.googleapis.com/auth/gmail.send"
            return False  # Overall check

        mock_has_required.side_effect = mock_check

        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.readonly"

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=[
                    "https://www.googleapis.com/auth/gmail.send",
                    "https://www.googleapis.com/auth/gmail.readonly"
                ],
                tool_name="gmail_tool"
            )

        detail = exc_info.value.detail
        assert "Missing:" in detail
        assert "https://www.googleapis.com/auth/gmail.send" in detail

    def test_validation_with_scope_hierarchy(self):
        """Test validation respects Google scope hierarchy."""
        # has_required_scopes should handle this, just verify it's called correctly
        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail"

        # If has_required_scopes is implemented correctly, this should pass
        # because gmail scope satisfies gmail.readonly
        with patch('seer.tools.google.validators.has_required_scopes') as mock:
            mock.return_value = True

            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.readonly"],
                tool_name="gmail_read"
            )

            # Verify has_required_scopes was called
            mock.assert_called()


@pytest.mark.unit
class TestValidationErrorDetails:
    """Test error message details and structure."""

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_error_includes_tool_name(self, mock_has_required):
        """Test that error message includes the tool name."""
        mock_has_required.return_value = False

        connection = Mock()
        connection.scopes = ""

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="my_custom_gmail_tool"
            )

        assert "my_custom_gmail_tool" in exc_info.value.detail

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_error_status_code_is_403(self, mock_has_required):
        """Test that validation failure returns 403 Forbidden."""
        mock_has_required.return_value = False

        connection = Mock()
        connection.scopes = ""

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="test_tool"
            )

        assert exc_info.value.status_code == 403

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_error_suggests_reconnection(self, mock_has_required):
        """Test that error message suggests reconnecting."""
        mock_has_required.return_value = False

        connection = Mock()
        connection.scopes = ""

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="test_tool"
            )

        assert "reconnect" in exc_info.value.detail.lower()

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_error_mentions_google_oauth_scopes(self, mock_has_required):
        """Test that error message mentions Google OAuth scopes."""
        mock_has_required.return_value = False

        connection = Mock()
        connection.scopes = ""

        with pytest.raises(HTTPException) as exc_info:
            validate_google_scopes(
                connection=connection,
                required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                tool_name="test_tool"
            )

        assert "Google OAuth scopes" in exc_info.value.detail


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_validation_with_many_scopes(self):
        """Test validation with many granted and required scopes."""
        connection = Mock()
        connection.scopes = " ".join([
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets",
        ])

        # Should pass if has_required_scopes is implemented correctly
        with patch('seer.tools.google.validators.has_required_scopes') as mock:
            mock.return_value = True

            validate_google_scopes(
                connection=connection,
                required_scopes=[
                    "https://www.googleapis.com/auth/gmail.send",
                    "https://www.googleapis.com/auth/spreadsheets",
                ],
                tool_name="multi_service_tool"
            )

    @patch('seer.tools.google.validators.has_required_scopes')
    def test_validation_with_single_required_scope(self, mock_has_required):
        """Test validation with just one required scope."""
        mock_has_required.return_value = True

        connection = Mock()
        connection.scopes = "https://www.googleapis.com/auth/gmail.send"

        validate_google_scopes(
            connection=connection,
            required_scopes=["https://www.googleapis.com/auth/gmail.send"],
            tool_name="gmail_send"
        )

        # Should call has_required_scopes
        mock_has_required.assert_called_once()

    def test_validation_preserves_scope_urls(self):
        """Test that full scope URLs are preserved in error messages."""
        connection = Mock()
        connection.scopes = ""

        with patch('seer.tools.google.validators.has_required_scopes') as mock:
            mock.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                validate_google_scopes(
                    connection=connection,
                    required_scopes=["https://www.googleapis.com/auth/gmail.send"],
                    tool_name="test_tool"
                )

            # Full URL should be in error message
            assert "googleapis.com" in exc_info.value.detail
