"""
Unit tests for markdown-to-HTML conversion in Gmail helpers.
"""
import pytest

from seer.tools.google.gmail.helpers import _build_mime_email, _markdown_to_html


class TestMarkdownToHtml:
    def test_bold_syntax(self):
        result = _markdown_to_html("**bold**")
        assert "<strong>bold</strong>" in result

    def test_header_syntax(self):
        result = _markdown_to_html("## Header")
        assert "<h2>" in result
        assert "Header" in result

    def test_unordered_list(self):
        result = _markdown_to_html("- item")
        assert "<ul>" in result
        assert "<li>" in result
        assert "item" in result

    def test_html_envelope(self):
        result = _markdown_to_html("Hello")
        assert "<!DOCTYPE html>" in result
        assert "<body" in result
        assert "font-family:Arial" in result

    def test_plain_text_still_wrapped(self):
        result = _markdown_to_html("Just plain text here.")
        assert "<!DOCTYPE html>" in result
        assert "Just plain text here." in result

    def test_empty_string(self):
        result = _markdown_to_html("")
        assert "<!DOCTYPE html>" in result


class TestBuildMimeEmailMarkdown:
    def _get_html_part(self, msg):
        """Extract the HTML alternative part from an EmailMessage."""
        for part in msg.iter_parts():
            if part.get_content_type() == "text/html":
                return part.get_payload(decode=True).decode()
        # For non-multipart messages, fall back
        return msg.get_payload(decode=True).decode() if msg.get_content_type() == "text/html" else ""

    def test_auto_generates_html_when_not_provided(self):
        msg = _build_mime_email(to=["test@example.com"], subject="Test", body_text="**bold text**")
        html_part = self._get_html_part(msg)
        assert "<strong>bold text</strong>" in html_part

    def test_does_not_override_explicit_body_html(self):
        custom_html = "<html><body><p>Custom HTML</p></body></html>"
        msg = _build_mime_email(
            to=["test@example.com"],
            subject="Test",
            body_text="**bold text**",
            body_html=custom_html,
        )
        html_part = self._get_html_part(msg)
        assert "Custom HTML" in html_part
        # The markdown conversion should NOT have been applied
        assert "<strong>bold text</strong>" not in html_part

    def test_plain_text_produces_valid_html(self):
        msg = _build_mime_email(to=["test@example.com"], subject="Test", body_text="Hello there")
        html_part = self._get_html_part(msg)
        assert "<!DOCTYPE html>" in html_part
        assert "Hello there" in html_part

    def test_message_is_multipart(self):
        msg = _build_mime_email(to=["test@example.com"], subject="Test", body_text="Hello")
        assert msg.is_multipart()

    def test_plain_text_part_preserved(self):
        msg = _build_mime_email(to=["test@example.com"], subject="Test", body_text="Plain text")
        plain_parts = [p for p in msg.iter_parts() if p.get_content_type() == "text/plain"]
        assert plain_parts, "Plain text part must be present"
        plain_content = plain_parts[0].get_payload(decode=True).decode()
        assert "Plain text" in plain_content
