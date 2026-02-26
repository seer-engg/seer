"""
Tests for shared workflow validation utilities.
"""

import pytest
from unittest.mock import patch, MagicMock

from seer.tools.workflow_validation import (
    ValidationError,
    ValidationResult,
    _get_attr_or_key,
    validate_tool_references,
    validate_trigger_references,
    validate_trigger_provider_configs,
    validate_tools_and_triggers,
    detect_extra_fields,
    format_validation_errors,
    run_full_validation,
)
from seer.tools.trigger_schema_fix import fix_trigger_event_schemas


@pytest.mark.unit
class TestValidationError:
    """Tests for ValidationError class."""

    def test_to_dict_without_hint(self):
        error = ValidationError("test_type", "Test message")
        result = error.to_dict()
        assert result == {
            "error_type": "test_type",
            "message": "Test message"
        }

    def test_to_dict_with_hint(self):
        error = ValidationError("test_type", "Test message", "Test hint")
        result = error.to_dict()
        assert result == {
            "error_type": "test_type",
            "message": "Test message",
            "hint": "Test hint"
        }


@pytest.mark.unit
class TestGetAttrOrKey:
    """Tests for _get_attr_or_key helper."""

    def test_gets_dict_key(self):
        obj = {"name": "test_value"}
        assert _get_attr_or_key(obj, "name") == "test_value"

    def test_gets_object_attribute(self):
        class MockObj:
            name = "test_value"
        obj = MockObj()
        assert _get_attr_or_key(obj, "name") == "test_value"

    def test_returns_default_for_missing_key(self):
        obj = {"other": "value"}
        assert _get_attr_or_key(obj, "name", "default") == "default"

    def test_returns_none_for_missing_without_default(self):
        obj = {"other": "value"}
        assert _get_attr_or_key(obj, "name") is None


@pytest.mark.unit
class TestValidateToolReferences:
    """Tests for validate_tool_references."""

    @patch("seer.tools.base.get_tool")
    def test_returns_empty_for_valid_tools(self, mock_get_tool):
        mock_get_tool.return_value = MagicMock()  # Tool exists
        spec = {
            "nodes": [
                {"type": "tool", "tool": "gmail_send_email"},
                {"type": "tool", "tool": "slack_post_message"},
            ]
        }
        errors = validate_tool_references(spec)
        assert errors == []

    @patch("seer.tools.base.get_tool")
    def test_returns_errors_for_missing_tools(self, mock_get_tool):
        mock_get_tool.return_value = None  # Tool doesn't exist
        spec = {
            "nodes": [
                {"type": "tool", "tool": "nonexistent_tool"},
            ]
        }
        errors = validate_tool_references(spec)
        assert len(errors) == 1
        assert "nonexistent_tool" in errors[0]
        assert "not found" in errors[0]

    @patch("seer.tools.base.get_tool")
    def test_skips_non_tool_nodes(self, mock_get_tool):
        mock_get_tool.return_value = None
        spec = {
            "nodes": [
                {"type": "condition", "expression": "true"},
                {"type": "output", "value": "result"},
            ]
        }
        errors = validate_tool_references(spec)
        assert errors == []
        mock_get_tool.assert_not_called()


@pytest.mark.unit
class TestValidateTriggerReferences:
    """Tests for validate_trigger_references."""

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_valid_triggers(self, mock_registry):
        mock_registry.maybe_get.return_value = MagicMock()  # Trigger exists
        mock_registry.all.return_value = []
        spec = {
            "triggers": [
                {"key": "gmail_new_email"},
            ]
        }
        errors = validate_trigger_references(spec)
        assert errors == []

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_errors_for_missing_triggers(self, mock_registry):
        mock_registry.maybe_get.return_value = None  # Trigger doesn't exist
        mock_trigger = MagicMock()
        mock_trigger.key = "valid_trigger"
        mock_registry.all.return_value = [mock_trigger]
        spec = {
            "triggers": [
                {"key": "nonexistent_trigger"},
            ]
        }
        errors = validate_trigger_references(spec)
        assert len(errors) == 1
        assert "nonexistent_trigger" in errors[0]
        assert "not found" in errors[0]

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_no_triggers(self, mock_registry):
        spec = {"triggers": []}
        errors = validate_trigger_references(spec)
        assert errors == []
        mock_registry.maybe_get.assert_not_called()


@pytest.mark.unit
class TestValidateTriggerProviderConfigs:
    """Tests for validate_trigger_provider_configs."""

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_no_triggers(self, mock_registry):
        """Test returns empty list when no triggers defined."""
        spec = {"triggers": []}
        errors = validate_trigger_provider_configs(spec)
        assert errors == []

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_no_provider_config(self, mock_registry):
        """Test returns empty list when trigger has no provider_config."""
        spec = {
            "triggers": [
                {"id": "t1", "key": "webhook.generic", "provider_config": {}}
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        assert errors == []

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_no_config_schema(self, mock_registry):
        """Test returns empty list when trigger has no config schema."""
        mock_definition = MagicMock()
        mock_definition.schemas.config = None
        mock_registry.maybe_get.return_value = mock_definition

        spec = {
            "triggers": [
                {"id": "t1", "key": "webhook.generic", "provider_config": {"any": "value"}}
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        assert errors == []

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_empty_for_valid_config(self, mock_registry):
        """Test returns empty list for valid provider_config."""
        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string"},
                "guild_id": {"type": "string"}
            },
            "required": ["channel_id", "guild_id"]
        }
        mock_registry.maybe_get.return_value = mock_definition

        spec = {
            "triggers": [
                {
                    "id": "t1",
                    "key": "poll.discord.message_received",
                    "provider_config": {"channel_id": "123", "guild_id": "456"}
                }
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        assert errors == []

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_error_for_missing_required_field(self, mock_registry):
        """Test returns error when required field is missing."""
        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
            "required": ["cron_expression", "timezone"]
        }
        mock_registry.maybe_get.return_value = mock_definition

        spec = {
            "triggers": [
                {
                    "id": "my_cron",
                    "key": "schedule.cron",
                    "provider_config": {"cron_expression": "0 * * * *"}  # Missing timezone
                }
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        assert len(errors) == 1
        assert "my_cron" in errors[0]
        assert "schedule.cron" in errors[0]
        assert "timezone" in errors[0]

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_error_for_wrong_type(self, mock_registry):
        """Test returns error when field has wrong type."""
        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer"}
            }
        }
        mock_registry.maybe_get.return_value = mock_definition

        spec = {
            "triggers": [
                {
                    "id": "t1",
                    "key": "poll.gmail.email_received",
                    "provider_config": {"max_results": "not_an_integer"}
                }
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        assert len(errors) == 1
        assert "t1" in errors[0]
        assert "not of type 'integer'" in errors[0]

    @patch("seer.core.registry.trigger_registry.trigger_registry")
    def test_returns_multiple_errors_for_multiple_issues(self, mock_registry):
        """Test returns multiple errors for multiple validation issues."""
        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "workspace_id": {"type": "string"},
                "channel_id": {"type": "string"}
            },
            "required": ["workspace_id", "channel_id"]
        }
        mock_registry.maybe_get.return_value = mock_definition

        spec = {
            "triggers": [
                {
                    "id": "slack_trigger",
                    "key": "poll.slack.message_received",
                    # Has some config but missing required fields
                    "provider_config": {"include_bot_messages": False}
                }
            ]
        }
        errors = validate_trigger_provider_configs(spec)
        # Should have errors for both missing required fields
        assert len(errors) == 2
        assert all("slack_trigger" in err for err in errors)


@pytest.mark.unit
class TestValidateToolsAndTriggers:
    """Tests for validate_tools_and_triggers convenience function."""

    @patch("seer.tools.workflow_validation.validate_trigger_references")
    @patch("seer.tools.workflow_validation.validate_tool_references")
    def test_combines_tool_and_trigger_errors(self, mock_tool_val, mock_trigger_val):
        mock_tool_val.return_value = ["Tool error"]
        mock_trigger_val.return_value = ["Trigger error"]
        spec = {"nodes": [], "triggers": []}
        errors = validate_tools_and_triggers(spec)
        assert errors == ["Tool error", "Trigger error"]


@pytest.mark.unit
class TestDetectExtraFields:
    """Tests for detect_extra_fields."""

    def test_returns_hint_for_extra_forbidden(self):
        spec = {"version": 2, "nodes": [], "edges": [], "invalid_field": "value"}
        error_msg = "extra_forbidden: 'invalid_field'"
        hint = detect_extra_fields(spec, error_msg)
        assert hint is not None
        assert "invalid_field" in hint
        assert "version, nodes, edges, triggers" in hint

    def test_returns_hint_for_extra_inputs(self):
        spec = {"version": 2, "nodes": [], "metadata": {}}
        error_msg = "Extra inputs are not permitted"
        hint = detect_extra_fields(spec, error_msg)
        assert hint is not None
        assert "metadata" in hint

    def test_returns_none_for_other_errors(self):
        spec = {"version": 2, "nodes": []}
        error_msg = "Some other validation error"
        hint = detect_extra_fields(spec, error_msg)
        assert hint is None

    def test_returns_none_when_no_invalid_fields(self):
        spec = {"version": 2, "nodes": [], "edges": [], "triggers": []}
        error_msg = "extra_forbidden somewhere"
        hint = detect_extra_fields(spec, error_msg)
        assert hint is None


@pytest.mark.unit
class TestFormatValidationErrors:
    """Tests for format_validation_errors."""

    def test_formats_errors_into_validation_error(self):
        errors = ["Error 1", "Error 2"]
        result = format_validation_errors(errors)
        assert isinstance(result, ValidationError)
        assert result.error_type == "reference_validation"
        assert "Error 1" in result.hint
        assert "Error 2" in result.hint

    def test_uses_custom_error_type(self):
        errors = ["Error"]
        result = format_validation_errors(errors, error_type="custom_type")
        assert result.error_type == "custom_type"


@pytest.mark.unit
class TestRunFullValidation:
    """Tests for the unified run_full_validation pipeline."""

    @pytest.mark.asyncio
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_returns_schema_error_on_invalid_spec(self, mock_parse):
        """Step 1 failure: Pydantic parse error yields schema_validation error."""
        from seer.core.errors import ValidationPhaseError
        mock_parse.side_effect = ValidationPhaseError("bad nodes")
        mock_user = MagicMock()

        result = await run_full_validation(mock_user, {"version": "2", "nodes": "bad"})

        assert result.success is False
        assert result.error.error_type == "schema_validation"
        assert "bad nodes" in result.error.message
        assert result.validated_spec is None
        assert result.schema_fixes == []

    @pytest.mark.asyncio
    @patch("seer.tools.workflow_validation.validate_tools_and_triggers")
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_returns_reference_error_for_bad_tools(self, mock_parse, mock_ref):
        """Step 2 failure: invalid tool reference yields reference_validation error."""
        mock_parse.return_value = MagicMock()
        mock_ref.return_value = ["Tool 'bad_tool' not found."]
        mock_user = MagicMock()

        result = await run_full_validation(mock_user, {"version": "2", "nodes": []})

        assert result.success is False
        assert result.error.error_type == "reference_validation"
        assert "bad_tool" in result.error.hint

    @pytest.mark.asyncio
    @patch("seer.tools.workflow_validation.validate_compilation")
    @patch("seer.tools.trigger_schema_fix.fix_trigger_event_schemas")
    @patch("seer.tools.workflow_validation.validate_tools_and_triggers")
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_trigger_auto_fix_applied_before_compilation(
        self, mock_parse, mock_ref, mock_fix, mock_compile
    ):
        """Step 3: trigger schemas are auto-fixed before compilation runs."""
        mock_parse.return_value = MagicMock()
        mock_ref.return_value = []
        fixed_dict = {"version": "2", "nodes": [], "triggers": [{"key": "k", "event_schema": {"fixed": True}}]}
        mock_fix.return_value = (fixed_dict, [{"trigger_id": "t1", "reason": "fixed"}])
        mock_compile.return_value = None
        mock_user = MagicMock()

        result = await run_full_validation(mock_user, {"version": "2", "nodes": []})

        assert result.success is True
        assert len(result.schema_fixes) == 1
        assert result.schema_fixes[0]["trigger_id"] == "t1"
        # Verify compilation was called with the FIXED spec
        mock_compile.assert_called_once()
        compile_spec_arg = mock_compile.call_args[0][1]
        assert compile_spec_arg is fixed_dict

    @pytest.mark.asyncio
    @patch("seer.tools.workflow_validation.validate_compilation")
    @patch("seer.tools.trigger_schema_fix.fix_trigger_event_schemas")
    @patch("seer.tools.workflow_validation.validate_tools_and_triggers")
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_compilation_error_still_includes_schema_fixes(
        self, mock_parse, mock_ref, mock_fix, mock_compile
    ):
        """Step 4 failure: compilation error includes any trigger auto-fixes that were applied."""
        mock_parse.return_value = MagicMock()
        mock_ref.return_value = []
        mock_fix.return_value = ({"version": "2", "nodes": []}, [{"trigger_id": "t1", "reason": "fixed"}])
        mock_compile.return_value = ValidationError("compilation", "graph cycle detected")
        mock_user = MagicMock()

        result = await run_full_validation(mock_user, {"version": "2", "nodes": []})

        assert result.success is False
        assert result.error.error_type == "compilation"
        assert len(result.schema_fixes) == 1

    @pytest.mark.asyncio
    @patch("seer.tools.workflow_validation.validate_compilation")
    @patch("seer.tools.trigger_schema_fix.fix_trigger_event_schemas")
    @patch("seer.tools.workflow_validation.validate_tools_and_triggers")
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_success_returns_validated_spec_and_fixes(
        self, mock_parse, mock_ref, mock_fix, mock_compile
    ):
        """Full success: returns validated_spec, fixed_spec_dict, and schema_fixes."""
        mock_spec = MagicMock()
        mock_parse.return_value = mock_spec
        mock_ref.return_value = []
        fixed_dict = {"version": "2", "nodes": []}
        mock_fix.return_value = (fixed_dict, [])
        mock_compile.return_value = None
        mock_user = MagicMock()

        result = await run_full_validation(mock_user, {"version": "2", "nodes": []})

        assert result.success is True
        assert result.validated_spec is mock_spec
        assert result.fixed_spec_dict is fixed_dict
        assert result.error is None
        assert result.schema_fixes == []

    @pytest.mark.asyncio
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_extra_fields_hint_on_schema_error(self, mock_parse):
        """Schema error with extra fields produces a helpful hint."""
        from seer.core.errors import ValidationPhaseError
        mock_parse.side_effect = ValidationPhaseError("extra_forbidden: 'metadata'")
        mock_user = MagicMock()

        result = await run_full_validation(
            mock_user,
            {"version": "2", "nodes": [], "metadata": {}}
        )

        assert result.success is False
        assert "metadata" in result.error.hint

    @pytest.mark.asyncio
    @patch("seer.tools.workflow_validation.validate_trigger_provider_configs")
    @patch("seer.tools.workflow_validation.validate_tools_and_triggers")
    @patch("seer.core.compiler.parse.parse_workflow_spec")
    async def test_returns_provider_config_error_for_invalid_trigger_config(
        self, mock_parse, mock_ref, mock_config
    ):
        """Step 2.5 failure: invalid trigger provider_config yields provider_config_validation error."""
        mock_parse.return_value = MagicMock()
        mock_ref.return_value = []  # No tool/trigger reference errors
        mock_config.return_value = ["Trigger 'my_cron' (schedule.cron): 'timezone' is a required property"]
        mock_user = MagicMock()

        result = await run_full_validation(
            mock_user,
            {"version": "2", "nodes": [], "triggers": [{"key": "schedule.cron"}]}
        )

        assert result.success is False
        assert result.error.error_type == "provider_config_validation"
        assert "timezone" in result.error.hint
        assert "my_cron" in result.error.hint


@pytest.mark.unit
class TestTriggerProviderConfigIntegration:
    """Integration tests for trigger provider_config validation with real registry."""

    def test_gmail_trigger_with_provider_connection_id_passes(self):
        """Test Gmail trigger with provider_connection_id is valid (no mocking)."""
        spec = {
            "version": "2",
            "triggers": [{
                "id": "gmail_trigger",
                "key": "poll.gmail.email_received",
                "mode": "polling",
                "provider_config": {
                    "provider_connection_id": 1
                }
            }],
            "nodes": [],
            "edges": []
        }

        errors = validate_trigger_provider_configs(spec)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_slack_trigger_with_provider_connection_id_passes(self):
        """Test Slack trigger with provider_connection_id is valid (no mocking)."""
        spec = {
            "version": "2",
            "triggers": [{
                "id": "slack_trigger",
                "key": "poll.slack.message_received",
                "mode": "polling",
                "provider_config": {
                    "workspace_id": "T12345",
                    "channel_id": "C67890",
                    "provider_connection_id": 42
                }
            }],
            "nodes": [],
            "edges": []
        }

        errors = validate_trigger_provider_configs(spec)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_discord_trigger_with_provider_connection_id_passes(self):
        """Test Discord trigger with provider_connection_id is valid (no mocking)."""
        spec = {
            "version": "2",
            "triggers": [{
                "id": "discord_trigger",
                "key": "poll.discord.message_received",
                "mode": "polling",
                "provider_config": {
                    "guild_id": "123456789",
                    "channel_id": "987654321",
                    "provider_connection_id": 7
                }
            }],
            "nodes": [],
            "edges": []
        }

        errors = validate_trigger_provider_configs(spec)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_google_calendar_trigger_with_provider_connection_id_passes(self):
        """Test Google Calendar trigger with provider_connection_id is valid (no mocking)."""
        spec = {
            "version": "2",
            "triggers": [{
                "id": "gcal_trigger",
                "key": "poll.google_calendar.event_changed",
                "mode": "polling",
                "provider_config": {
                    "calendar_id": "primary",
                    "provider_connection_id": 3
                }
            }],
            "nodes": [],
            "edges": []
        }

        errors = validate_trigger_provider_configs(spec)
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_provider_connection_id_must_be_integer_integration(self):
        """Test that provider_connection_id validation rejects non-integer values."""
        spec = {
            "version": "2",
            "triggers": [{
                "id": "gmail_trigger",
                "key": "poll.gmail.email_received",
                "mode": "polling",
                "provider_config": {
                    "provider_connection_id": "not-an-integer"  # Wrong type
                }
            }],
            "nodes": [],
            "edges": []
        }

        errors = validate_trigger_provider_configs(spec)
        assert len(errors) == 1
        assert "not of type 'integer'" in errors[0]


@pytest.mark.unit
class TestFixTriggerEventSchemas:
    """Tests for fix_trigger_event_schemas and custom data property merging."""

    def test_fix_trigger_preserves_custom_data_properties(self):
        """Auto-fix should merge custom data.properties into canonical schema."""
        spec_dict = {
            "triggers": [{
                "id": "form_input",
                "key": "form.hosted",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "tone": {"type": "string"}
                            }
                        }
                    }
                }
            }]
        }

        fixed_spec, fixes = fix_trigger_event_schemas(spec_dict)

        # Should have applied a fix
        assert len(fixes) == 1
        assert "custom data fields preserved" in fixes[0]["reason"]

        # Custom properties should be merged
        data_schema = fixed_spec["triggers"][0]["event_schema"]["properties"]["data"]
        assert data_schema.get("additionalProperties") is True  # Still flexible
        assert "topic" in data_schema.get("properties", {})
        assert "tone" in data_schema.get("properties", {})

    def test_fix_trigger_replaces_entirely_when_no_custom_properties(self):
        """Auto-fix should replace entirely when spec has no custom data.properties."""
        spec_dict = {
            "triggers": [{
                "id": "webhook_trigger",
                "key": "webhook.generic",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "wrong_field": {"type": "string"}
                    }
                }
            }]
        }

        fixed_spec, fixes = fix_trigger_event_schemas(spec_dict)

        # Should have applied a fix
        assert len(fixes) == 1
        assert "replaced with canonical schema" in fixes[0]["reason"]

    def test_fix_trigger_no_change_when_schema_matches(self):
        """No fix applied when spec schema matches canonical exactly."""
        # First, get the canonical schema
        from seer.core.registry.trigger_registry import trigger_registry

        canonical_def = trigger_registry.get("form.hosted")
        canonical_schema = canonical_def.schemas.event

        spec_dict = {
            "triggers": [{
                "id": "form_input",
                "key": "form.hosted",
                "event_schema": canonical_schema  # Exact match
            }]
        }

        fixed_spec, fixes = fix_trigger_event_schemas(spec_dict)

        # No fixes should be applied
        assert len(fixes) == 0

    def test_fix_trigger_available_fields_include_custom_properties(self):
        """Feedback should show custom fields in available_fields."""
        spec_dict = {
            "triggers": [{
                "id": "content_form",
                "key": "form.hosted",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "tone": {"type": "string"},
                                "key_points": {"type": "string"}
                            }
                        }
                    }
                }
            }]
        }

        _, fixes = fix_trigger_event_schemas(spec_dict)

        # Custom fields should appear in available_fields
        assert "topic" in fixes[0]["available_fields"]["data"]
        assert "tone" in fixes[0]["available_fields"]["data"]
        assert "key_points" in fixes[0]["available_fields"]["data"]

        # Example expressions should use custom fields
        assert "${content_form.data.topic}" in fixes[0]["example_expressions"]
