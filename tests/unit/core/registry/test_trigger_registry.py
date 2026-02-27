"""Tests for trigger registry config schemas."""

import pytest
from jsonschema import Draft7Validator

from seer.core.registry.trigger_registry import (
    _gmail_email_received_config_schema,
    _slack_message_received_config_schema,
    _google_calendar_event_changed_config_schema,
    _discord_message_received_config_schema,
)


class TestOAuthTriggerConfigSchemas:
    """Test that OAuth triggers accept provider_connection_id."""

    def test_gmail_accepts_provider_connection_id(self):
        """Test that Gmail config schema accepts provider_connection_id."""
        schema = _gmail_email_received_config_schema()
        validator = Draft7Validator(schema)

        # Gmail has no required fields
        config = {"provider_connection_id": 1}
        errors = list(validator.iter_errors(config))

        assert not errors, f"Schema rejected provider_connection_id: {errors}"

    def test_slack_accepts_provider_connection_id(self):
        """Test that Slack config schema accepts provider_connection_id."""
        schema = _slack_message_received_config_schema()
        validator = Draft7Validator(schema)

        # Slack requires workspace_id and channel_id
        config = {
            "workspace_id": "T12345",
            "channel_id": "C67890",
            "provider_connection_id": 1,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Schema rejected provider_connection_id: {errors}"

    def test_google_calendar_accepts_provider_connection_id(self):
        """Test that Google Calendar config schema accepts provider_connection_id."""
        schema = _google_calendar_event_changed_config_schema()
        validator = Draft7Validator(schema)

        # Google Calendar has no required fields
        config = {"provider_connection_id": 1}
        errors = list(validator.iter_errors(config))

        assert not errors, f"Schema rejected provider_connection_id: {errors}"

    def test_discord_accepts_provider_connection_id(self):
        """Test that Discord config schema accepts provider_connection_id."""
        schema = _discord_message_received_config_schema()
        validator = Draft7Validator(schema)

        # Discord requires guild_id and channel_id
        config = {
            "guild_id": "123456789",
            "channel_id": "987654321",
            "provider_connection_id": 1,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Schema rejected provider_connection_id: {errors}"

    @pytest.mark.parametrize(
        "schema_fn,base_config",
        [
            (_gmail_email_received_config_schema, {}),
            (_slack_message_received_config_schema, {"workspace_id": "T1", "channel_id": "C1"}),
            (_google_calendar_event_changed_config_schema, {}),
            (_discord_message_received_config_schema, {"guild_id": "G1", "channel_id": "C1"}),
        ],
        ids=["gmail", "slack", "google_calendar", "discord"],
    )
    def test_provider_connection_id_must_be_integer(self, schema_fn, base_config):
        """Test that provider_connection_id must be an integer."""
        schema = schema_fn()
        validator = Draft7Validator(schema)

        config = {**base_config, "provider_connection_id": "not-an-int"}
        errors = list(validator.iter_errors(config))

        # Should have exactly one error about type
        type_errors = [e for e in errors if "not of type" in str(e.message).lower()]
        assert len(type_errors) == 1, f"Expected one type error, got: {errors}"

    @pytest.mark.parametrize(
        "schema_fn,base_config",
        [
            (_gmail_email_received_config_schema, {}),
            (_slack_message_received_config_schema, {"workspace_id": "T1", "channel_id": "C1"}),
            (_google_calendar_event_changed_config_schema, {}),
            (_discord_message_received_config_schema, {"guild_id": "G1", "channel_id": "C1"}),
        ],
        ids=["gmail", "slack", "google_calendar", "discord"],
    )
    def test_provider_connection_id_is_optional(self, schema_fn, base_config):
        """Test that provider_connection_id is optional (not required)."""
        schema = schema_fn()
        validator = Draft7Validator(schema)

        # Config without provider_connection_id should be valid
        errors = list(validator.iter_errors(base_config))

        # Filter out errors related to required fields other than provider_connection_id
        connection_id_required_errors = [
            e for e in errors if "provider_connection_id" in str(e.message)
        ]
        assert not connection_id_required_errors, (
            f"provider_connection_id should be optional but got: {connection_id_required_errors}"
        )

    def test_gmail_config_with_all_fields(self):
        """Test Gmail config schema accepts all valid fields including provider_connection_id."""
        schema = _gmail_email_received_config_schema()
        validator = Draft7Validator(schema)

        config = {
            "label_ids": ["INBOX", "UNREAD"],
            "query": "is:unread",
            "max_results": 10,
            "overlap_ms": 60000,
            "provider_connection_id": 42,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Gmail config with all fields should be valid: {errors}"

    def test_slack_config_with_all_fields(self):
        """Test Slack config schema accepts all valid fields including provider_connection_id."""
        schema = _slack_message_received_config_schema()
        validator = Draft7Validator(schema)

        config = {
            "workspace_id": "T12345",
            "channel_id": "C67890",
            "include_bot_messages": True,
            "only_app_mentions": False,
            "max_results": 25,
            "provider_connection_id": 99,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Slack config with all fields should be valid: {errors}"

    def test_discord_config_with_all_fields(self):
        """Test Discord config schema accepts all valid fields including provider_connection_id."""
        schema = _discord_message_received_config_schema()
        validator = Draft7Validator(schema)

        config = {
            "guild_id": "123456789",
            "channel_id": "987654321",
            "include_bot_messages": False,
            "only_mentions": True,
            "max_results": 50,
            "provider_connection_id": 7,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Discord config with all fields should be valid: {errors}"

    def test_google_calendar_config_with_all_fields(self):
        """Test Google Calendar config schema accepts all valid fields including provider_connection_id."""
        schema = _google_calendar_event_changed_config_schema()
        validator = Draft7Validator(schema)

        config = {
            "calendar_id": "primary",
            "max_results": 25,
            "provider_connection_id": 3,
        }
        errors = list(validator.iter_errors(config))

        assert not errors, f"Google Calendar config with all fields should be valid: {errors}"
