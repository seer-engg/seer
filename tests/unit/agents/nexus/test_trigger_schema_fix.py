"""
Unit tests for trigger event_schema auto-fix functionality in workflow_tools.

Tests the _schemas_differ, _extract_available_fields, _build_example_expressions,
and fix_trigger_event_schemas functions.
"""
import pytest
from copy import deepcopy
from unittest.mock import patch, MagicMock

from seer.tools.trigger_schema_fix import (
    _schemas_differ,
    _extract_available_fields,
    _build_example_expressions,
    _detect_misplaced_properties,
    fix_trigger_event_schemas,
)


@pytest.mark.unit
class TestSchemasDiffer:
    """Test _schemas_differ function."""

    def test_empty_dict_returns_true(self):
        """Empty spec schema should trigger replacement."""
        canonical = {"type": "object", "properties": {"data": {}}}
        assert _schemas_differ({}, canonical) is True

    def test_none_returns_true(self):
        """None spec schema should trigger replacement."""
        canonical = {"type": "object", "properties": {"data": {}}}
        assert _schemas_differ(None, canonical) is True

    def test_matching_schemas_returns_false(self):
        """Matching schemas should not trigger replacement."""
        schema = {"type": "object", "properties": {"data": {"type": "string"}}}
        assert _schemas_differ(schema, schema) is False

    def test_different_schemas_returns_true(self):
        """Different schemas should trigger replacement."""
        spec_schema = {"type": "object", "properties": {"message_id": {}}}
        canonical_schema = {"type": "object", "properties": {"data": {"properties": {"message_id": {}}}}}
        assert _schemas_differ(spec_schema, canonical_schema) is True


@pytest.mark.unit
class TestExtractAvailableFields:
    """Test _extract_available_fields function."""

    def test_extracts_envelope_and_data_fields(self):
        """Should extract field names from envelope and data properties."""
        canonical_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                        "subject": {"type": "string"},
                        "from": {"type": "object"}
                    }
                }
            }
        }

        result = _extract_available_fields(canonical_schema)

        assert "envelope" in result
        assert "data" in result
        assert "id" in result["envelope"]
        assert "trigger_key" in result["envelope"]
        assert "data" in result["envelope"]
        assert "message_id" in result["data"]
        assert "subject" in result["data"]
        assert "from" in result["data"]

    def test_empty_schema_returns_empty_lists(self):
        """Empty schema should return empty field lists."""
        result = _extract_available_fields({})
        assert result == {"envelope": [], "data": []}

    def test_schema_without_data_returns_empty_data_fields(self):
        """Schema without data property should return empty data fields."""
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        result = _extract_available_fields(schema)
        assert result["envelope"] == ["id"]
        assert result["data"] == []


@pytest.mark.unit
class TestBuildExampleExpressions:
    """Test _build_example_expressions function."""

    def test_builds_simple_field_expressions(self):
        """Should build ${trigger.data.field} expressions for simple fields."""
        canonical_schema = {
            "properties": {
                "data": {
                    "properties": {
                        "message_id": {"type": "string"},
                        "subject": {"type": "string"}
                    }
                }
            }
        }

        result = _build_example_expressions("new_email", canonical_schema)

        assert "${new_email.data.message_id}" in result
        assert "${new_email.data.subject}" in result

    def test_builds_nested_field_expressions(self):
        """Should build ${trigger.data.field.nested} for object fields."""
        canonical_schema = {
            "properties": {
                "data": {
                    "properties": {
                        "from": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "name": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

        result = _build_example_expressions("trigger", canonical_schema)

        # Should have a nested expression like ${trigger.data.from.email}
        assert any(".from." in expr for expr in result)

    def test_limits_to_three_examples(self):
        """Should return at most 3 example expressions."""
        canonical_schema = {
            "properties": {
                "data": {
                    "properties": {
                        "field1": {"type": "string"},
                        "field2": {"type": "string"},
                        "field3": {"type": "string"},
                        "field4": {"type": "string"},
                        "field5": {"type": "string"},
                    }
                }
            }
        }

        result = _build_example_expressions("t", canonical_schema)

        assert len(result) <= 3

    def test_empty_schema_returns_empty_list(self):
        """Empty schema should return empty list."""
        result = _build_example_expressions("trigger", {})
        assert result == []


@pytest.mark.unit
class TestFixTriggerEventSchemas:
    """Test fix_trigger_event_schemas function."""

    @pytest.fixture
    def canonical_gmail_schema(self):
        """Sample canonical schema for Gmail trigger."""
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "message_id": {"type": "string"},
                        "thread_id": {"type": "string"},
                        "subject": {"type": "string"},
                        "from": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "name": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

    @pytest.fixture
    def mock_trigger_definition(self, canonical_gmail_schema):
        """Create a mock trigger definition."""
        mock_def = MagicMock()
        mock_def.schemas.event = canonical_gmail_schema
        return mock_def

    def test_fixes_empty_event_schema(self, mock_trigger_definition, canonical_gmail_schema):
        """Should replace empty event_schema with canonical."""
        spec_dict = {
            "triggers": [
                {
                    "id": "new_email",
                    "key": "poll.gmail.email_received",
                    "mode": "polling",
                    "event_schema": {}
                }
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_trigger_definition

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 1
        assert fixes[0]["trigger_id"] == "new_email"
        assert fixes[0]["trigger_key"] == "poll.gmail.email_received"
        assert "Empty event_schema" in fixes[0]["reason"]
        assert result_spec["triggers"][0]["event_schema"] == canonical_gmail_schema

    def test_fixes_incorrect_event_schema(self, mock_trigger_definition, canonical_gmail_schema):
        """Should replace incorrect event_schema with canonical."""
        incorrect_schema = {
            "type": "object",
            "properties": {
                "message_id": {"type": "string"},  # Missing data envelope!
                "from": {"type": "object"}
            }
        }

        spec_dict = {
            "triggers": [
                {
                    "id": "email_trigger",
                    "key": "poll.gmail.email_received",
                    "mode": "polling",
                    "event_schema": incorrect_schema
                }
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_trigger_definition

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 1
        assert "Incorrect event_schema" in fixes[0]["reason"]
        assert result_spec["triggers"][0]["event_schema"] == canonical_gmail_schema

    def test_does_not_modify_correct_schema(self, mock_trigger_definition, canonical_gmail_schema):
        """Should not modify schema that matches canonical."""
        spec_dict = {
            "triggers": [
                {
                    "id": "email_trigger",
                    "key": "poll.gmail.email_received",
                    "mode": "polling",
                    "event_schema": deepcopy(canonical_gmail_schema)
                }
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_trigger_definition

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 0
        assert result_spec["triggers"][0]["event_schema"] == canonical_gmail_schema

    def test_handles_multiple_triggers(self, canonical_gmail_schema):
        """Should process each trigger independently."""
        mock_gmail = MagicMock()
        mock_gmail.schemas.event = canonical_gmail_schema

        mock_webhook = MagicMock()
        mock_webhook.schemas.event = {"type": "object", "properties": {"payload": {}}}

        spec_dict = {
            "triggers": [
                {"id": "t1", "key": "poll.gmail.email_received", "event_schema": {}},
                {"id": "t2", "key": "webhook.generic", "event_schema": {}}
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.side_effect = lambda key: {
                "poll.gmail.email_received": mock_gmail,
                "webhook.generic": mock_webhook
            }.get(key)

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 2
        assert fixes[0]["trigger_id"] == "t1"
        assert fixes[1]["trigger_id"] == "t2"

    def test_skips_unknown_triggers(self):
        """Should skip triggers not found in registry."""
        spec_dict = {
            "triggers": [
                {"id": "t1", "key": "unknown.trigger", "event_schema": {}}
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = None

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 0
        assert result_spec["triggers"][0]["event_schema"] == {}

    def test_skips_triggers_without_canonical_schema(self):
        """Should skip triggers where canonical has no event schema."""
        mock_def = MagicMock()
        mock_def.schemas.event = None

        spec_dict = {
            "triggers": [
                {"id": "t1", "key": "some.trigger", "event_schema": {}}
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_def

            result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 0

    def test_handles_spec_without_triggers(self):
        """Should handle spec with no triggers gracefully."""
        spec_dict = {"nodes": [], "edges": []}

        result_spec, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 0
        assert result_spec == spec_dict

    def test_fix_feedback_includes_available_fields(self, mock_trigger_definition, canonical_gmail_schema):
        """Fix feedback should include available_fields for LLM guidance."""
        spec_dict = {
            "triggers": [
                {"id": "t1", "key": "poll.gmail.email_received", "event_schema": {}}
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_trigger_definition

            _, fixes = fix_trigger_event_schemas(spec_dict)

        assert "available_fields" in fixes[0]
        assert "envelope" in fixes[0]["available_fields"]
        assert "data" in fixes[0]["available_fields"]

    def test_fix_feedback_includes_example_expressions(self, mock_trigger_definition, canonical_gmail_schema):
        """Fix feedback should include example_expressions for LLM guidance."""
        spec_dict = {
            "triggers": [
                {"id": "email", "key": "poll.gmail.email_received", "event_schema": {}}
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_trigger_definition

            _, fixes = fix_trigger_event_schemas(spec_dict)

        assert "example_expressions" in fixes[0]
        # Should contain expressions using the trigger id
        assert any("email.data." in expr for expr in fixes[0]["example_expressions"])


@pytest.mark.unit
class TestDetectMisplacedProperties:
    """Test _detect_misplaced_properties function."""

    def test_detects_root_level_custom_properties(self):
        """Should detect properties at root level that belong in data."""
        spec_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "context": {"type": "string"},
                "style": {"type": "string"},
            }
        }
        canonical_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {"type": "object", "additionalProperties": True},
            }
        }

        result = _detect_misplaced_properties(spec_schema, canonical_schema)

        assert "topic" in result
        assert "context" in result
        assert "style" in result

    def test_ignores_envelope_fields(self):
        """Should not flag standard envelope fields as misplaced."""
        spec_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {"type": "object"},
                "topic": {"type": "string"},  # This is misplaced
            }
        }
        canonical_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {"type": "object"},
            }
        }

        result = _detect_misplaced_properties(spec_schema, canonical_schema)

        assert "id" not in result
        assert "trigger_key" not in result
        assert "data" not in result
        assert "topic" in result

    def test_returns_empty_for_none_schema(self):
        """Should return empty list for None spec schema."""
        result = _detect_misplaced_properties(None, {"properties": {}})
        assert result == []

    def test_returns_empty_for_empty_schema(self):
        """Should return empty list for empty spec schema."""
        result = _detect_misplaced_properties({}, {"properties": {}})
        assert result == []

    def test_returns_sorted_list(self):
        """Should return sorted list of misplaced properties."""
        spec_schema = {
            "properties": {
                "zebra": {"type": "string"},
                "alpha": {"type": "string"},
                "middle": {"type": "string"},
            }
        }
        canonical_schema = {"properties": {"data": {}}}

        result = _detect_misplaced_properties(spec_schema, canonical_schema)

        assert result == ["alpha", "middle", "zebra"]


@pytest.mark.unit
class TestFixTriggerEventSchemasStrippedProperties:
    """Test fix_trigger_event_schemas with misplaced properties detection."""

    def test_fix_includes_stripped_properties_warning(self):
        """Fix record should include warning about stripped root-level properties."""
        canonical_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "trigger_key": {"type": "string"},
                "data": {"type": "object", "additionalProperties": True},
            }
        }

        mock_def = MagicMock()
        mock_def.schemas.event = canonical_schema

        # Schema with properties at WRONG level (root instead of data)
        incorrect_schema = {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "context": {"type": "string"},
            }
        }

        spec_dict = {
            "triggers": [
                {
                    "id": "form_input",
                    "key": "form.hosted",
                    "event_schema": incorrect_schema
                }
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_def

            _, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 1
        fix = fixes[0]

        # Should include stripped properties
        assert "stripped_properties" in fix
        assert "topic" in fix["stripped_properties"]
        assert "context" in fix["stripped_properties"]

        # Should include warning
        assert "warning" in fix
        assert "root level" in fix["warning"]
        assert "data" in fix["warning"]

        # Should include correct structure hint
        assert "correct_structure_hint" in fix
        hint = fix["correct_structure_hint"]
        assert "event_schema" in hint
        assert "properties" in hint["event_schema"]
        assert "data" in hint["event_schema"]["properties"]
        # Hint should include the actual stripped property names
        hint_data_props = hint["event_schema"]["properties"]["data"]["properties"]
        assert "topic" in hint_data_props
        assert "context" in hint_data_props

    def test_no_warning_when_no_misplaced_properties(self):
        """Fix record should NOT include warning when properties are correctly placed."""
        canonical_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "data": {
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {}
                },
            }
        }

        mock_def = MagicMock()
        mock_def.schemas.event = canonical_schema

        # Empty schema (no misplaced properties)
        spec_dict = {
            "triggers": [
                {
                    "id": "t1",
                    "key": "form.hosted",
                    "event_schema": {}
                }
            ]
        }

        with patch("seer.tools.trigger_schema_fix.trigger_registry") as mock_registry:
            mock_registry.maybe_get.return_value = mock_def

            _, fixes = fix_trigger_event_schemas(spec_dict)

        assert len(fixes) == 1
        fix = fixes[0]

        # Should NOT have stripped_properties or warning
        assert fix.get("stripped_properties") is None or fix.get("stripped_properties") == []
        assert fix.get("warning") is None
        assert fix.get("correct_structure_hint") is None
