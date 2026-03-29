"""Unit tests for E2B result serialization."""
from __future__ import annotations

import pytest

from seer.tools.e2b.base import serialize_e2b_result
from tests.unit.tools.e2b.conftest import MockResult


@pytest.mark.unit
class TestSerializeE2BResult:
    """Test serialize_e2b_result helper."""

    def test_text_result(self):
        """Text attribute should map to text/plain."""
        result = MockResult(text="hello world")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/plain"
        assert serialized["data"] == "hello world"

    def test_json_result(self):
        """JSON attribute should map to application/json."""
        result = MockResult(json={"key": "value"})
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "application/json"
        assert serialized["data"] == {"key": "value"}

    def test_png_result(self):
        """PNG attribute should map to image/png."""
        result = MockResult(png="iVBORw0KGgo...")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "image/png"
        assert serialized["data"] == "iVBORw0KGgo..."

    def test_html_result(self):
        """HTML attribute should map to text/html."""
        result = MockResult(html="<div>hello</div>")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/html"
        assert serialized["data"] == "<div>hello</div>"

    def test_markdown_result(self):
        """Markdown attribute should map to text/markdown."""
        result = MockResult(markdown="# Title")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/markdown"
        assert serialized["data"] == "# Title"

    def test_svg_result(self):
        """SVG attribute should map to image/svg+xml."""
        result = MockResult(svg="<svg></svg>")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "image/svg+xml"

    def test_jpeg_result(self):
        """JPEG attribute should map to image/jpeg."""
        result = MockResult(jpeg="/9j/4AAQ...")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "image/jpeg"

    def test_priority_text_over_json(self):
        """Text should be preferred over JSON when both are present."""
        result = MockResult(text="plain text", json={"key": "value"})
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/plain"
        assert serialized["data"] == "plain text"

    def test_priority_json_over_html(self):
        """JSON should be preferred over HTML when both are present."""
        result = MockResult(json={"key": "value"}, html="<div>hello</div>")
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "application/json"

    def test_fallback_no_known_attributes(self):
        """Should fall back to str(result) when no known attributes are set."""
        result = MockResult()  # All E2B attributes are None
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/plain"
        assert isinstance(serialized["data"], str)

    def test_plain_object_fallback(self):
        """Should handle arbitrary objects without E2B attributes."""

        class BareObject:
            pass

        result = BareObject()
        serialized = serialize_e2b_result(result)
        assert serialized["type"] == "text/plain"
        assert isinstance(serialized["data"], str)
