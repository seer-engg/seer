"""
Unit tests for api.workflows.services.triggers pure logic.

Tests validation, URL building, expression evaluation, and event generation.
Heavy mock tests for DB operations have been moved to E2E tests.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# =============================================================================
# Validate Filters Payload Tests
# =============================================================================


class TestValidateFiltersPayload:
    """Tests for _validate_filters_payload function."""

    def test_validate_filters_empty_filters_passes(self):
        """Test empty filters always pass validation."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {"type": "object"}

        # Should not raise
        _validate_filters_payload({}, mock_definition)
        _validate_filters_payload(None, mock_definition)

    def test_validate_filters_no_schema_passes(self):
        """Test validation passes when no filter schema defined."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = None

        # Should not raise
        _validate_filters_payload({"any": "value"}, mock_definition)

    def test_validate_filters_valid_filters_pass(self):
        """Test valid filters pass schema validation."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {
            "type": "object",
            "properties": {
                "status": {"type": "string"}
            }
        }

        # Should not raise
        _validate_filters_payload({"status": "active"}, mock_definition)

    def test_validate_filters_invalid_raises_problem(self):
        """Test invalid filters raise validation problem."""
        from seer.api.workflows.services.triggers import _validate_filters_payload

        mock_definition = MagicMock()
        mock_definition.schemas.filter = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Validation failed")

            with pytest.raises(Exception, match="Validation failed"):
                _validate_filters_payload({"count": "not_an_integer"}, mock_definition)

        mock_raise.assert_called_once()
        call_kwargs = mock_raise.call_args[1]
        assert call_kwargs["status"] == 400


# =============================================================================
# Validate Provider Config Tests
# =============================================================================


class TestValidateProviderConfig:
    """Tests for _validate_provider_config function."""

    def test_validate_provider_config_empty_passes(self):
        """Test empty provider_config always passes validation."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {"type": "object", "required": ["channel_id"]}

        # Should not raise - empty config skips validation
        _validate_provider_config({}, mock_definition)
        _validate_provider_config(None, mock_definition)

    def test_validate_provider_config_no_schema_passes(self):
        """Test validation passes when no config schema defined."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = None

        # Should not raise
        _validate_provider_config({"any": "value"}, mock_definition)

    def test_validate_provider_config_valid_passes(self):
        """Test valid provider_config passes schema validation."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string"},
                "guild_id": {"type": "string"}
            },
            "required": ["channel_id", "guild_id"]
        }

        # Should not raise
        _validate_provider_config(
            {"channel_id": "123", "guild_id": "456"},
            mock_definition
        )

    def test_validate_provider_config_invalid_raises_problem(self):
        """Test invalid provider_config raises validation problem."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
            "required": ["cron_expression", "timezone"]
        }

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Validation failed")

            # Missing required "timezone" field
            with pytest.raises(Exception, match="Validation failed"):
                _validate_provider_config({"cron_expression": "0 * * * *"}, mock_definition)

        mock_raise.assert_called_once()
        call_kwargs = mock_raise.call_args[1]
        assert call_kwargs["status"] == 400
        assert "Invalid trigger configuration" in call_kwargs["title"]

    def test_validate_provider_config_wrong_type_raises_problem(self):
        """Test provider_config with wrong type raises validation problem."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer"}
            }
        }

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Validation failed")

            # max_results should be integer, not string
            with pytest.raises(Exception, match="Validation failed"):
                _validate_provider_config({"max_results": "not_an_integer"}, mock_definition)

        mock_raise.assert_called_once()

    def test_validate_provider_config_autocorrects_timezone_abbreviation(self):
        """Test that timezone abbreviations like 'CST' are auto-corrected to IANA names."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
        }

        config = {"cron_expression": "0 * * * *", "timezone": "CST"}
        _validate_provider_config(config, mock_definition)
        assert config["timezone"] == "America/Chicago"

    def test_validate_provider_config_autocorrects_pst(self):
        """Test that PST is auto-corrected to America/Los_Angeles."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
        }

        config = {"cron_expression": "0 * * * *", "timezone": "PST"}
        _validate_provider_config(config, mock_definition)
        assert config["timezone"] == "America/Los_Angeles"

    def test_validate_provider_config_rejects_unknown_timezone(self):
        """Test that truly invalid timezone values are still rejected."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
        }

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Validation failed")

            with pytest.raises(Exception, match="Validation failed"):
                _validate_provider_config(
                    {"cron_expression": "0 * * * *", "timezone": "NotATimezone"},
                    mock_definition,
                )

        mock_raise.assert_called_once()
        call_kwargs = mock_raise.call_args[1]
        assert call_kwargs["status"] == 400

    def test_validate_provider_config_accepts_valid_timezone(self):
        """Test that valid IANA timezone names are accepted."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
        }

        # Should not raise
        _validate_provider_config(
            {"cron_expression": "0 * * * *", "timezone": "America/New_York"},
            mock_definition,
        )

    def test_validate_provider_config_strips_timezone_whitespace(self):
        """Test that trailing whitespace in timezone is handled."""
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "properties": {
                "cron_expression": {"type": "string"},
                "timezone": {"type": "string"}
            },
        }

        # 'America/Chicago ' with trailing space should pass after strip
        _validate_provider_config(
            {"cron_expression": "0 * * * *", "timezone": "America/Chicago "},
            mock_definition,
        )

    def test_validate_provider_config_excludes_provider_connection_id(self):
        """Test that provider_connection_id is excluded from schema validation.

        Regression test: provider_connection_id is an infrastructure field that specifies
        which OAuth connection to use. It should not be validated against the trigger's
        config schema (which has additionalProperties: false for most triggers).
        """
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "additionalProperties": False,  # Strict schema - no extra properties allowed
            "properties": {
                "label_ids": {"type": "array", "items": {"type": "string"}},
                "query": {"type": "string"},
            }
        }

        # Should NOT raise even though provider_connection_id is not in schema
        # because it's an infrastructure field that should be excluded from validation
        _validate_provider_config(
            {"provider_connection_id": 1, "query": "is:unread"},
            mock_definition
        )

    def test_timezone_enum_values_are_valid_pytz_timezones(self):
        """Test that all timezone enum values in the schema are valid pytz timezones."""
        import pytz
        from seer.core.registry.trigger_registry import trigger_registry

        definition = trigger_registry.maybe_get("schedule.cron")
        assert definition is not None
        schema = definition.schemas.config
        tz_enum = schema["properties"]["timezone"]["enum"]
        for tz_name in tz_enum:
            pytz.timezone(tz_name)  # Raises if invalid

    def test_validate_provider_config_only_provider_connection_id_passes(self):
        """Test that provider_config with ONLY provider_connection_id passes validation.

        When provider_config only contains the infrastructure field, validation should pass
        since there's nothing else to validate.
        """
        from seer.api.workflows.services.triggers import _validate_provider_config

        mock_definition = MagicMock()
        mock_definition.schemas.config = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "some_field": {"type": "string"},
            }
        }

        # Should not raise - only infrastructure field present
        _validate_provider_config({"provider_connection_id": 42}, mock_definition)


# =============================================================================
# Is Expression Tests
# =============================================================================


class TestIsExpression:
    """Tests for _is_expression function."""

    def test_is_expression_valid_expression(self):
        """Test valid expression pattern detection."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("${event.data.message}") is True
        assert _is_expression("  ${event.data.value}  ") is True  # With whitespace

    def test_is_expression_invalid_patterns(self):
        """Test non-expression values."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("plain_string") is False
        assert _is_expression("${incomplete") is False
        assert _is_expression("incomplete}") is False
        assert _is_expression(123) is False
        assert _is_expression(None) is False

    def test_is_expression_nested_braces(self):
        """Test expression with nested content."""
        from seer.api.workflows.services.triggers import _is_expression

        assert _is_expression("${event.data.nested.path}") is True


# =============================================================================
# Build Webhook URL Tests
# =============================================================================


class TestBuildWebhookUrl:
    """Tests for _build_webhook_url function."""

    def test_build_webhook_url_generic(self):
        """Test building URL for generic webhook using slug."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url("test_slug_abc123", "webhook.generic")
        assert result == "/api/v1/webhooks/generic/test_slug_abc123"

    def test_build_webhook_url_supabase(self):
        """Test building URL for Supabase webhook using slug."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url("supabase_slug_xyz789", "webhook.supabase.db_changes")
        assert result == "/api/v1/webhooks/generic/supabase_slug_xyz789"

    def test_build_webhook_url_unknown_type(self):
        """Test building URL for unknown trigger type returns None."""
        from seer.api.workflows.services.triggers import _build_webhook_url

        result = _build_webhook_url("any_slug", "schedule.cron")
        assert result is None


# =============================================================================
# Validate Form Suffix Tests
# =============================================================================


class TestValidateFormSuffix:
    """Tests for _validate_form_suffix function."""

    def test_validate_form_suffix_valid(self):
        """Test valid form suffix passes validation."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        # Should not raise
        _validate_form_suffix("contact-form")
        _validate_form_suffix("my-form-123")
        _validate_form_suffix("form")
        _validate_form_suffix(None)

    def test_validate_form_suffix_invalid_chars_raises(self):
        """Test form suffix with invalid characters raises problem."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Invalid suffix")

            with pytest.raises(Exception, match="Invalid suffix"):
                _validate_form_suffix("has spaces!")

    def test_validate_form_suffix_reserved_raises(self):
        """Test reserved form suffix raises problem."""
        from seer.api.workflows.services.triggers import _validate_form_suffix

        with patch("seer.api.workflows.services.triggers._raise_problem") as mock_raise:
            mock_raise.side_effect = Exception("Reserved suffix")

            with pytest.raises(Exception, match="Reserved suffix"):
                _validate_form_suffix("admin")


# =============================================================================
# Extract Event Path Tests
# =============================================================================


class TestExtractEventPath:
    """Tests for _extract_event_path function."""

    def test_extract_event_path_simple(self):
        """Test extracting simple event path."""
        from seer.api.workflows.services.triggers import _extract_event_path

        result = _extract_event_path("${event.message}")
        assert result == ["message"]

    def test_extract_event_path_nested(self):
        """Test extracting nested event path."""
        from seer.api.workflows.services.triggers import _extract_event_path

        result = _extract_event_path("${event.data.user.name}")
        assert result == ["data", "user", "name"]

    def test_extract_event_path_non_event_raises(self):
        """Test non-event reference raises ValueError."""
        from seer.api.workflows.services.triggers import _extract_event_path

        with pytest.raises(ValueError, match="must reference event"):
            _extract_event_path("${nodes.task1.output}")

    def test_extract_event_path_too_short_raises(self):
        """Test path with no property raises ValueError."""
        from seer.api.workflows.services.triggers import _extract_event_path

        # "${event}" doesn't start with "event." so raises "must reference event.*"
        with pytest.raises(ValueError, match="must reference event"):
            _extract_event_path("${event}")


# =============================================================================
# Evaluate Bindings Tests
# =============================================================================


class TestEvaluateBindings:
    """Tests for _evaluate_bindings function."""

    def test_evaluate_bindings_expression(self):
        """Test evaluating expression bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {"user_email": "${event.data.email}"}
        event_payload = {"data": {"email": "test@example.com"}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {"user_email": "test@example.com"}

    def test_evaluate_bindings_literal(self):
        """Test evaluating literal bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {"static_value": "hello"}
        event_payload = {"data": {}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {"static_value": "hello"}

    def test_evaluate_bindings_mixed(self):
        """Test evaluating mixed expression and literal bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        bindings = {
            "email": "${event.data.email}",
            "source": "webhook"
        }
        event_payload = {"data": {"email": "user@example.com"}}

        result = _evaluate_bindings(bindings, event_payload)
        assert result == {
            "email": "user@example.com",
            "source": "webhook"
        }

    def test_evaluate_bindings_empty(self):
        """Test evaluating empty/None bindings."""
        from seer.api.workflows.services.triggers import _evaluate_bindings

        assert _evaluate_bindings({}, {"data": {}}) == {}
        assert _evaluate_bindings(None, {"data": {}}) == {}


# =============================================================================
# Generate Cron Event Tests
# =============================================================================


class TestGenerateCronEvent:
    """Tests for generate_cron_event function."""

    def test_generate_cron_event_basic(self):
        """Test generating a basic cron event envelope."""
        from seer.api.workflows.services.triggers import generate_cron_event

        result = generate_cron_event("trigger_123", {})

        # Verify envelope structure
        assert "id" in result
        assert result["id"].startswith("evt_")
        assert result["trigger_id"] == "trigger_123"
        assert result["trigger_key"] == "schedule.cron"
        assert result["title"] == "Manual Trigger"
        assert result["provider"] == "schedule"
        assert result["account_id"] is None
        assert "occurred_at" in result
        assert "received_at" in result

        # Verify payload defaults
        assert result["data"]["cron_expression"] == "* * * * *"
        assert result["data"]["timezone"] == "UTC"
        assert result["data"]["manual"] is True

    def test_generate_cron_event_with_config(self):
        """Test generating a cron event with provider config."""
        from seer.api.workflows.services.triggers import generate_cron_event

        provider_config = {
            "cron_expression": "0 9 * * MON-FRI",
            "timezone": "America/New_York"
        }

        result = generate_cron_event("trigger_456", provider_config)

        assert result["trigger_id"] == "trigger_456"
        assert result["data"]["cron_expression"] == "0 9 * * MON-FRI"
        assert result["data"]["timezone"] == "America/New_York"
        assert result["data"]["manual"] is True

    def test_generate_cron_event_timestamps(self):
        """Test that cron event has valid ISO timestamps."""
        from seer.api.workflows.services.triggers import generate_cron_event

        result = generate_cron_event("trigger_789", {})

        # Verify timestamps are valid ISO format
        scheduled_time = datetime.fromisoformat(result["data"]["scheduled_time"])
        actual_time = datetime.fromisoformat(result["data"]["actual_time"])
        occurred_at = datetime.fromisoformat(result["occurred_at"])
        received_at = datetime.fromisoformat(result["received_at"])

        # All timestamps should be timezone-aware
        assert all(t.tzinfo is not None for t in [scheduled_time, actual_time, occurred_at, received_at])

    def test_generate_cron_event_unique_ids(self):
        """Test that each generated event has a unique ID."""
        from seer.api.workflows.services.triggers import generate_cron_event

        event1 = generate_cron_event("trigger_1", {})
        event2 = generate_cron_event("trigger_1", {})

        assert event1["id"] != event2["id"]


# =============================================================================
# Cursor Reset on Re-enable Tests
# =============================================================================


class TestApplySubscriptionUpdatesReenableCursorReset:
    """Tests for _apply_subscription_updates cursor reset when re-enabling."""

    def test_reenable_polling_subscription_resets_cursor(self):
        """Re-enabling a disabled polling subscription should clear poll_cursor_json
        to force bootstrap_cursor() on next poll, preventing catchup storms.
        next_poll_at must be set to current time (not None) to satisfy NOT NULL constraint."""
        from seer.api.workflows.services.triggers import _apply_subscription_updates

        subscription = MagicMock()
        subscription.enabled = False
        subscription.is_polling = True
        subscription.poll_cursor_json = {"last_execution_utc": "2026-03-27T18:30:00+00:00"}
        subscription.next_poll_at = datetime(2026, 3, 27, 18, 45)
        subscription.poll_status = "disabled"
        subscription.poll_error_json = {"reason": "oauth_error"}
        subscription.trigger_key = "schedule.cron"

        payload = MagicMock()
        payload.enabled = True
        payload.filters = None
        payload.provider_connection_id = None
        payload.provider_config = None

        definition = MagicMock()

        _apply_subscription_updates(subscription, payload, definition)

        assert subscription.poll_cursor_json is None
        # next_poll_at must never be None (NOT NULL column) — should be reset to ~now
        assert subscription.next_poll_at is not None
        assert isinstance(subscription.next_poll_at, datetime)
        assert subscription.poll_status == "ok"
        assert subscription.poll_error_json is None
        assert subscription.enabled is True

    def test_reenable_non_polling_subscription_preserves_cursor(self):
        """Re-enabling a non-polling subscription (e.g. webhook) should NOT reset cursor."""
        from seer.api.workflows.services.triggers import _apply_subscription_updates

        subscription = MagicMock()
        subscription.enabled = False
        subscription.is_polling = False
        subscription.trigger_key = "webhook.generic"

        payload = MagicMock()
        payload.enabled = True
        payload.filters = None
        payload.provider_connection_id = None
        payload.provider_config = None

        definition = MagicMock()
        original_cursor = {"some": "cursor"}
        subscription.poll_cursor_json = original_cursor

        _apply_subscription_updates(subscription, payload, definition)

        assert subscription.poll_cursor_json == original_cursor
        assert subscription.enabled is True

    def test_disable_does_not_reset_cursor(self):
        """Disabling a subscription should NOT reset the cursor."""
        from seer.api.workflows.services.triggers import _apply_subscription_updates

        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = True
        subscription.trigger_key = "schedule.cron"

        payload = MagicMock()
        payload.enabled = False
        payload.filters = None
        payload.provider_connection_id = None
        payload.provider_config = None

        definition = MagicMock()
        original_cursor = {"last_execution_utc": "2026-03-30T12:00:00+00:00"}
        subscription.poll_cursor_json = original_cursor

        _apply_subscription_updates(subscription, payload, definition)

        assert subscription.poll_cursor_json == original_cursor
        assert subscription.enabled is False

    def test_already_enabled_does_not_reset_cursor(self):
        """Setting enabled=True on an already-enabled subscription should NOT reset cursor."""
        from seer.api.workflows.services.triggers import _apply_subscription_updates

        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = True
        subscription.trigger_key = "schedule.cron"

        payload = MagicMock()
        payload.enabled = True
        payload.filters = None
        payload.provider_connection_id = None
        payload.provider_config = None

        definition = MagicMock()
        original_cursor = {"last_execution_utc": "2026-03-30T12:00:00+00:00"}
        subscription.poll_cursor_json = original_cursor

        _apply_subscription_updates(subscription, payload, definition)

        # Cursor should be preserved — subscription was already enabled
        assert subscription.poll_cursor_json == original_cursor


class TestUpdateExistingSubscriptionReenableCursorReset:
    """Tests for _update_existing_subscription cursor/status reset when re-enabling."""

    async def test_reenable_disabled_polling_subscription_resets_state(self):
        """Re-enabling a disabled polling subscription via sync should reset cursor,
        next_poll_at (to non-None), poll_status, and poll_error_json."""
        from seer.api.workflows.services.triggers import _update_existing_subscription

        subscription = MagicMock()
        subscription.enabled = False
        subscription.is_polling = True
        subscription.poll_cursor_json = {"last_execution_utc": "2026-03-27T18:30:00+00:00"}
        subscription.next_poll_at = datetime(2026, 3, 27, 18, 45, tzinfo=timezone.utc)
        subscription.poll_status = "disabled"
        subscription.poll_error_json = {"reason": "adapter_permanent_error", "detail": {"status": 403}}
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.webhook_slug = None
        subscription.provider_connection_id = "conn-1"
        subscription.form_suffix = None
        subscription.save = AsyncMock()

        trigger_spec = MagicMock()
        trigger_spec.key = "poll.gmail.email_received"
        trigger_spec.filters = {}
        trigger_spec.provider_config = {"provider_connection_id": "conn-1"}

        definition = MagicMock()
        definition.title = "Gmail Trigger"

        await _update_existing_subscription(
            subscription,
            trigger_spec,
            definition,
            webhook_slug=None,
            adjusted_interval=60,
            skip_validation=True,
        )

        assert subscription.poll_cursor_json is None
        assert subscription.next_poll_at is not None
        assert isinstance(subscription.next_poll_at, datetime)
        assert subscription.poll_status == "ok"
        assert subscription.poll_error_json is None
        assert subscription.enabled is True
        subscription.save.assert_awaited_once()

    async def test_already_enabled_subscription_preserves_poll_state_when_config_unchanged(self):
        """Syncing an already-enabled subscription with unchanged config should NOT reset cursor."""
        from seer.api.workflows.services.triggers import _update_existing_subscription

        provider_config = {"provider_connection_id": "conn-1"}

        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = True
        subscription.poll_cursor_json = {"last_execution_utc": "2026-04-07T10:00:00+00:00"}
        original_next_poll = datetime(2026, 4, 7, 10, 1, tzinfo=timezone.utc)
        subscription.next_poll_at = original_next_poll
        subscription.poll_status = "ok"
        subscription.poll_error_json = None
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.webhook_slug = None
        subscription.provider_connection_id = "conn-1"
        subscription.provider_config = provider_config
        subscription.form_suffix = None
        subscription.save = AsyncMock()

        trigger_spec = MagicMock()
        trigger_spec.key = "poll.gmail.email_received"
        trigger_spec.filters = {}
        trigger_spec.provider_config = provider_config

        definition = MagicMock()
        definition.title = "Gmail Trigger"

        await _update_existing_subscription(
            subscription,
            trigger_spec,
            definition,
            webhook_slug=None,
            adjusted_interval=60,
            skip_validation=True,
        )

        # Cursor and poll state should be preserved when config hasn't changed
        assert subscription.poll_cursor_json == {"last_execution_utc": "2026-04-07T10:00:00+00:00"}
        assert subscription.poll_status == "ok"

    async def test_already_enabled_subscription_resets_cursor_when_config_changes(self):
        """Syncing an already-enabled subscription with changed config should reset cursor."""
        from seer.api.workflows.services.triggers import _update_existing_subscription

        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = True
        subscription.poll_cursor_json = {"cron_expression": "0 7 * * *", "last_execution_utc": "2026-04-07T12:00:00+00:00"}
        subscription.next_poll_at = datetime(2026, 4, 7, 12, 0, tzinfo=timezone.utc)
        subscription.poll_status = "ok"
        subscription.poll_error_json = None
        subscription.trigger_key = "schedule.cron"
        subscription.webhook_slug = None
        subscription.provider_connection_id = None
        subscription.provider_config = {"cron_expression": "0 7 * * *", "timezone": "America/Chicago"}
        subscription.form_suffix = None
        subscription.save = AsyncMock()

        trigger_spec = MagicMock()
        trigger_spec.key = "schedule.cron"
        trigger_spec.filters = {}
        trigger_spec.provider_config = {"cron_expression": "0 6 * * *", "timezone": "America/Chicago"}

        definition = MagicMock()
        definition.title = "Cron Trigger"

        await _update_existing_subscription(
            subscription,
            trigger_spec,
            definition,
            webhook_slug=None,
            adjusted_interval=60,
            skip_validation=True,
        )

        # Cursor should be reset when provider_config changes
        assert subscription.poll_cursor_json is None
        assert subscription.next_poll_at is not None
        assert isinstance(subscription.next_poll_at, datetime)
        assert subscription.poll_status == "ok"
        assert subscription.poll_error_json is None


# =============================================================================
# sync_trigger_subscriptions slug regeneration (regression guard for DB wipe)
# =============================================================================


@pytest.mark.asyncio
class TestSyncTriggerSubscriptionsSlugRegeneration:
    """Guard against the DB-wipe scenario: existing webhook.generic subscriptions
    with NULL webhook_slug must get a fresh slug after sync_trigger_subscriptions."""

    async def test_null_slug_regenerated_for_existing_webhook_generic_subscription(self):
        """If an existing webhook.generic subscription has webhook_slug=None
        (e.g. after a DB column reset), sync_trigger_subscriptions must assign
        a valid slug and the safety-net pass must persist it."""
        from unittest.mock import AsyncMock, MagicMock, patch, call
        from seer.api.workflows.services.triggers import (
            _update_existing_subscription,
            _should_emit_webhook_url,
            _generate_webhook_slug,
        )

        # Verify the helper functions behave as expected
        assert _should_emit_webhook_url("webhook.generic") is True
        assert _should_emit_webhook_url("form.hosted") is False
        assert _should_emit_webhook_url("webhook.twilio.whatsapp") is False

        # Simulate an existing subscription whose slug was wiped
        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = False
        subscription.webhook_slug = None  # ← wiped by DB fix
        subscription.form_suffix = None
        subscription.provider_connection_id = None
        subscription.save = AsyncMock()

        trigger_spec = MagicMock()
        trigger_spec.key = "webhook.generic"
        trigger_spec.filters = {}
        trigger_spec.provider_config = {}

        definition = MagicMock()
        definition.title = "Generic Webhook"

        # previous_slug=None → slug gets generated before calling _update_existing_subscription
        previous_slug = subscription.webhook_slug  # None
        webhook_slug = previous_slug
        if _should_emit_webhook_url(trigger_spec.key) and not webhook_slug:
            webhook_slug = _generate_webhook_slug()

        assert webhook_slug is not None
        assert len(webhook_slug) > 10  # token_urlsafe(32) is 43 chars

        await _update_existing_subscription(
            subscription,
            trigger_spec,
            definition,
            webhook_slug=webhook_slug,
            adjusted_interval=60,
            skip_validation=True,
        )

        # The slug must have been set on the subscription object
        assert subscription.webhook_slug == webhook_slug
        subscription.save.assert_awaited_once()

    async def test_existing_valid_slug_preserved(self):
        """If an existing subscription already has a valid webhook_slug,
        sync must preserve it (not generate a new one)."""
        from seer.api.workflows.services.triggers import (
            _update_existing_subscription,
            _should_emit_webhook_url,
            _generate_webhook_slug,
        )

        existing_slug = "existing-valid-slug-abc123"

        subscription = MagicMock()
        subscription.enabled = True
        subscription.is_polling = False
        subscription.webhook_slug = existing_slug
        subscription.form_suffix = None
        subscription.provider_connection_id = None
        subscription.save = AsyncMock()

        trigger_spec = MagicMock()
        trigger_spec.key = "webhook.generic"
        trigger_spec.filters = {}
        trigger_spec.provider_config = {}

        definition = MagicMock()
        definition.title = "Generic Webhook"

        # Mirrors the logic in sync_trigger_subscriptions
        previous_slug = subscription.webhook_slug
        webhook_slug = previous_slug
        if _should_emit_webhook_url(trigger_spec.key) and not webhook_slug:
            webhook_slug = _generate_webhook_slug()

        # Slug should be preserved, not regenerated
        assert webhook_slug == existing_slug

        await _update_existing_subscription(
            subscription,
            trigger_spec,
            definition,
            webhook_slug=webhook_slug,
            adjusted_interval=60,
            skip_validation=True,
        )

        assert subscription.webhook_slug == existing_slug
