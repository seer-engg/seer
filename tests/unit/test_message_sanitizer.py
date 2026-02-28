"""Unit tests for message_sanitizer utilities."""
import pytest

from seer.utilities.message_sanitizer import sanitize_tool_call_id, sanitize_messages_tool_call_ids


class TestSanitizeToolCallId:
    """Tests for sanitize_tool_call_id function."""

    @pytest.mark.parametrize("input_id,expected", [
        # IDs with leading/trailing whitespace - should be stripped
        (" list_available_tools:8", "list_available_tools:8"),
        (" search_tools:5", "search_tools:5"),
        (" functions.list_available_triggers:13", "functions.list_available_triggers:13"),
        ("  spaces_around  ", "spaces_around"),
        # IDs without whitespace - should pass through unchanged
        ("functions.search_tools:9", "functions.search_tools:9"),
        ("valid_id-123", "valid_id-123"),
        ("tool_call_abc123", "tool_call_abc123"),
        ("call-12345", "call-12345"),
        ("multiple...dots", "multiple...dots"),
        ("colons:in:id", "colons:in:id"),
        ("mixed.chars:and-stuff", "mixed.chars:and-stuff"),
    ])
    def test_sanitize_strips_whitespace_only(self, input_id: str, expected: str):
        """Test that only leading/trailing whitespace is stripped."""
        result = sanitize_tool_call_id(input_id)
        assert result == expected

    def test_empty_string_generates_uuid(self):
        """Test that empty string generates a UUID-based ID."""
        result = sanitize_tool_call_id("")
        assert result.startswith("tool_call_")
        assert len(result) == len("tool_call_") + 8

    def test_none_generates_uuid(self):
        """Test that None generates a UUID-based ID."""
        result = sanitize_tool_call_id(None)
        assert result.startswith("tool_call_")
        assert len(result) == len("tool_call_") + 8

    def test_whitespace_only_generates_uuid(self):
        """Test that whitespace-only string generates a UUID-based ID."""
        result = sanitize_tool_call_id("   ")
        assert result.startswith("tool_call_")

    def test_preserves_special_chars(self):
        """Test that special chars (colons, periods) are preserved."""
        result = sanitize_tool_call_id("functions.tool:123")
        assert result == "functions.tool:123"


class TestSanitizeMessagesToolCallIds:
    """Tests for sanitize_messages_tool_call_ids function."""

    def test_sanitizes_dict_tool_calls(self):
        """Test sanitization of dict-based tool_calls."""
        class MockAIMessage:
            def __init__(self):
                self.tool_calls = [
                    {"id": " list_tools:8", "name": "list_tools"},
                    {"id": "functions.search:9", "name": "search"},
                ]

        messages = [MockAIMessage()]
        result = sanitize_messages_tool_call_ids(messages)

        assert result[0].tool_calls[0]["id"] == "list_tools:8"  # Whitespace stripped
        assert result[0].tool_calls[1]["id"] == "functions.search:9"  # No change (no whitespace)

    def test_sanitizes_tool_message_id(self):
        """Test sanitization of ToolMessage.tool_call_id."""
        class MockToolMessage:
            def __init__(self):
                self.tool_call_id = " response:123"

        messages = [MockToolMessage()]
        result = sanitize_messages_tool_call_ids(messages)

        assert result[0].tool_call_id == "response:123"  # Whitespace stripped

    def test_handles_messages_without_tool_calls(self):
        """Test that messages without tool_calls are handled gracefully."""
        class MockHumanMessage:
            def __init__(self):
                self.content = "Hello"

        messages = [MockHumanMessage()]
        result = sanitize_messages_tool_call_ids(messages)

        assert len(result) == 1
        assert result[0].content == "Hello"

    def test_handles_empty_tool_calls_list(self):
        """Test handling of empty tool_calls list."""
        class MockAIMessage:
            def __init__(self):
                self.tool_calls = []

        messages = [MockAIMessage()]
        result = sanitize_messages_tool_call_ids(messages)

        assert result[0].tool_calls == []

    def test_handles_mixed_message_types(self):
        """Test handling of mixed message types in a conversation."""
        class MockHumanMessage:
            def __init__(self):
                self.content = "Hello"

        class MockAIMessage:
            def __init__(self):
                self.tool_calls = [{"id": " tool:1", "name": "tool"}]

        class MockToolMessage:
            def __init__(self):
                self.tool_call_id = " tool:1"
                self.content = "Result"

        messages = [MockHumanMessage(), MockAIMessage(), MockToolMessage()]
        result = sanitize_messages_tool_call_ids(messages)

        assert len(result) == 3
        assert result[1].tool_calls[0]["id"] == "tool:1"  # Whitespace stripped
        assert result[2].tool_call_id == "tool:1"  # Whitespace stripped
