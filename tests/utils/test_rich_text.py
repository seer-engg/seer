"""Tests for rich text converter."""

import pytest

from seer.utils.rich_text import Platform, RichTextConverter, convert_rich_text


class TestRichTextConverter:
    """Tests for RichTextConverter class."""

    @pytest.fixture
    def converter(self) -> RichTextConverter:
        return RichTextConverter()

    # --- HTML (Email/Calendar) Tests ---

    def test_to_html_basic(self, converter: RichTextConverter):
        """Test basic HTML conversion."""
        result = converter.to_html("Hello **world**")
        assert "<strong>world</strong>" in result
        assert "<!DOCTYPE html>" in result

    def test_to_html_italic(self, converter: RichTextConverter):
        """Test italic conversion to HTML."""
        result = converter.to_html("Hello *world*")
        assert "<em>world</em>" in result

    def test_to_html_link(self, converter: RichTextConverter):
        """Test link conversion to HTML."""
        result = converter.to_html("Check [this](https://example.com)")
        assert '<a href="https://example.com">this</a>' in result

    def test_to_html_list(self, converter: RichTextConverter):
        """Test list conversion to HTML."""
        result = converter.to_html("- item 1\n- item 2")
        assert "<li>item 1</li>" in result
        assert "<li>item 2</li>" in result

    def test_to_html_code(self, converter: RichTextConverter):
        """Test inline code conversion to HTML."""
        result = converter.to_html("Use `code` here")
        assert "<code>code</code>" in result

    def test_to_html_empty(self, converter: RichTextConverter):
        """Test empty content."""
        result = converter.convert("", Platform.EMAIL)
        assert result == ""

    # --- Slack mrkdwn Tests ---

    def test_to_slack_bold(self, converter: RichTextConverter):
        """Test bold conversion for Slack."""
        result = converter.to_slack_mrkdwn("Hello **world**")
        assert result == "Hello *world*"

    def test_to_slack_bold_underscore(self, converter: RichTextConverter):
        """Test bold with underscores for Slack."""
        result = converter.to_slack_mrkdwn("Hello __world__")
        assert result == "Hello *world*"

    def test_to_slack_italic(self, converter: RichTextConverter):
        """Test italic conversion for Slack."""
        # Single asterisks should become underscores
        result = converter.to_slack_mrkdwn("Hello *world*")
        assert result == "Hello _world_"

    def test_to_slack_strikethrough(self, converter: RichTextConverter):
        """Test strikethrough conversion for Slack."""
        result = converter.to_slack_mrkdwn("Hello ~~world~~")
        assert result == "Hello ~world~"

    def test_to_slack_link(self, converter: RichTextConverter):
        """Test link conversion for Slack."""
        result = converter.to_slack_mrkdwn("Check [this](https://example.com)")
        assert result == "Check <https://example.com|this>"

    def test_to_slack_code(self, converter: RichTextConverter):
        """Test code stays unchanged for Slack."""
        result = converter.to_slack_mrkdwn("Use `code` here")
        assert result == "Use `code` here"

    def test_to_slack_blockquote(self, converter: RichTextConverter):
        """Test blockquote stays unchanged for Slack."""
        result = converter.to_slack_mrkdwn("> quoted text")
        assert result == "> quoted text"

    def test_to_slack_complex(self, converter: RichTextConverter):
        """Test complex Slack conversion."""
        input_text = "Hello **bold** and *italic* with [link](https://x.com)"
        result = converter.to_slack_mrkdwn(input_text)
        assert "*bold*" in result
        assert "_italic_" in result
        assert "<https://x.com|link>" in result

    # --- Discord Tests ---

    def test_to_discord_bold(self, converter: RichTextConverter):
        """Test bold stays as-is for Discord."""
        result = converter.to_discord("Hello **world**")
        assert result == "Hello **world**"

    def test_to_discord_underline_conversion(self, converter: RichTextConverter):
        """Test __text__ converts to bold for Discord (since Discord uses __ for underline)."""
        result = converter.to_discord("Hello __world__")
        assert result == "Hello **world**"

    def test_to_discord_italic(self, converter: RichTextConverter):
        """Test italic stays as-is for Discord."""
        result = converter.to_discord("Hello *world*")
        assert result == "Hello *world*"

    def test_to_discord_strikethrough(self, converter: RichTextConverter):
        """Test strikethrough stays as-is for Discord."""
        result = converter.to_discord("Hello ~~world~~")
        assert result == "Hello ~~world~~"

    def test_to_discord_code(self, converter: RichTextConverter):
        """Test code stays as-is for Discord."""
        result = converter.to_discord("Use `code` here")
        assert result == "Use `code` here"

    # --- Platform enum tests ---

    def test_convert_with_platform_enum(self, converter: RichTextConverter):
        """Test convert method with Platform enum."""
        result = converter.convert("**bold**", Platform.SLACK)
        assert result == "*bold*"

    def test_convert_email_platform(self, converter: RichTextConverter):
        """Test EMAIL platform produces HTML."""
        result = converter.convert("**bold**", Platform.EMAIL)
        assert "<strong>bold</strong>" in result

    def test_convert_calendar_platform(self, converter: RichTextConverter):
        """Test CALENDAR platform produces HTML."""
        result = converter.convert("**bold**", Platform.CALENDAR)
        assert "<strong>bold</strong>" in result


class TestConvertRichTextFunction:
    """Tests for the convenience function."""

    def test_with_platform_enum(self):
        """Test with Platform enum."""
        result = convert_rich_text("**bold**", Platform.SLACK)
        assert result == "*bold*"

    def test_with_platform_string(self):
        """Test with platform string."""
        result = convert_rich_text("**bold**", "slack")
        assert result == "*bold*"

    def test_with_invalid_platform_string(self):
        """Test with invalid platform string raises error."""
        with pytest.raises(ValueError):
            convert_rich_text("**bold**", "invalid_platform")
