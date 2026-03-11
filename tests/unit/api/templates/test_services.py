"""
Unit tests for template services.

Tests:
- Config field extraction from workflow specs
- Placeholder resolution in specs
- Template listing and filtering
- Template instantiation logic
- Requirements checking
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.api.templates.services import (
    extract_config_fields,
    _resolve_placeholders,
    _apply_provider_connections,
)


# =============================================================================
# Config Field Extraction Tests
# =============================================================================


@pytest.mark.unit
def test_extract_config_fields_simple():
    """Test extraction of simple config fields."""
    spec = {
        "nodes": [
            {
                "id": "node1",
                "type": "tool",
                "tool": "email.send",
                "inputs": {
                    "to": "${config.recipient_email}",
                    "subject": "Hello ${config.name}",
                },
            }
        ]
    }

    fields = extract_config_fields(spec)

    assert len(fields) == 2
    field_names = {f.name for f in fields}
    assert "recipient_email" in field_names
    assert "name" in field_names

    # Check label generation
    for field in fields:
        if field.name == "recipient_email":
            assert field.label == "Recipient Email"
        if field.name == "name":
            assert field.label == "Name"


@pytest.mark.unit
def test_extract_config_fields_nested():
    """Test extraction from deeply nested structures."""
    spec = {
        "nodes": [
            {
                "id": "node1",
                "inputs": {
                    "nested": {
                        "deep": {
                            "value": "${config.deep_value}",
                        }
                    }
                },
            }
        ],
        "metadata": {
            "config": {
                "setting": "${config.setting_value}",
            }
        },
    }

    fields = extract_config_fields(spec)

    field_names = {f.name for f in fields}
    assert "deep_value" in field_names
    assert "setting_value" in field_names


@pytest.mark.unit
def test_extract_config_fields_in_arrays():
    """Test extraction from arrays."""
    spec = {
        "nodes": [
            {"inputs": {"value": "${config.value_1}"}},
            {"inputs": {"value": "${config.value_2}"}},
        ],
        "tags": ["${config.tag_1}", "static_tag"],
    }

    fields = extract_config_fields(spec)

    field_names = {f.name for f in fields}
    assert "value_1" in field_names
    assert "value_2" in field_names
    assert "tag_1" in field_names


@pytest.mark.unit
def test_extract_config_fields_no_duplicates():
    """Test that duplicate config fields are not repeated."""
    spec = {
        "nodes": [
            {"inputs": {"to": "${config.email}"}},
            {"inputs": {"cc": "${config.email}"}},
            {"inputs": {"bcc": "${config.email}"}},
        ]
    }

    fields = extract_config_fields(spec)

    # Should only have one "email" field
    assert len(fields) == 1
    assert fields[0].name == "email"


@pytest.mark.unit
def test_extract_config_fields_empty_spec():
    """Test extraction from empty spec."""
    spec = {}
    fields = extract_config_fields(spec)
    assert fields == []


@pytest.mark.unit
def test_extract_config_fields_no_config_placeholders():
    """Test extraction when no config placeholders exist."""
    spec = {
        "nodes": [
            {
                "inputs": {
                    "value": "${trigger.data.value}",  # Not a config placeholder
                    "static": "plain text",
                }
            }
        ]
    }

    fields = extract_config_fields(spec)
    assert fields == []


# =============================================================================
# Placeholder Resolution Tests
# =============================================================================


@pytest.mark.unit
def test_resolve_placeholders_simple():
    """Test simple placeholder resolution."""
    spec = {
        "inputs": {
            "to": "${config.email}",
            "name": "${config.name}",
        }
    }

    config = {"email": "test@example.com", "name": "John"}

    result = _resolve_placeholders(spec, config)

    assert result["inputs"]["to"] == "test@example.com"
    assert result["inputs"]["name"] == "John"


@pytest.mark.unit
def test_resolve_placeholders_inline():
    """Test placeholder resolution within longer strings."""
    spec = {
        "inputs": {
            "message": "Hello ${config.name}, your order ${config.order_id} is ready!",
        }
    }

    config = {"name": "Alice", "order_id": "12345"}

    result = _resolve_placeholders(spec, config)

    assert result["inputs"]["message"] == "Hello Alice, your order 12345 is ready!"


@pytest.mark.unit
def test_resolve_placeholders_nested():
    """Test placeholder resolution in nested structures."""
    spec = {
        "level1": {
            "level2": {
                "level3": "${config.value}",
            }
        }
    }

    config = {"value": "deep_value"}

    result = _resolve_placeholders(spec, config)

    assert result["level1"]["level2"]["level3"] == "deep_value"


@pytest.mark.unit
def test_resolve_placeholders_in_arrays():
    """Test placeholder resolution in arrays."""
    spec = {
        "recipients": [
            "${config.email_1}",
            "${config.email_2}",
            "static@example.com",
        ]
    }

    config = {"email_1": "a@example.com", "email_2": "b@example.com"}

    result = _resolve_placeholders(spec, config)

    assert result["recipients"][0] == "a@example.com"
    assert result["recipients"][1] == "b@example.com"
    assert result["recipients"][2] == "static@example.com"


@pytest.mark.unit
def test_resolve_placeholders_missing_value():
    """Test that missing config values keep the placeholder."""
    spec = {
        "inputs": {
            "value": "${config.missing_key}",
        }
    }

    config = {}  # No values provided

    result = _resolve_placeholders(spec, config)

    # Placeholder should remain unchanged
    assert result["inputs"]["value"] == "${config.missing_key}"


@pytest.mark.unit
def test_resolve_placeholders_preserves_non_config():
    """Test that non-config placeholders are preserved."""
    spec = {
        "inputs": {
            "trigger_value": "${trigger.data.value}",
            "node_value": "${node1.output.result}",
            "config_value": "${config.setting}",
        }
    }

    config = {"setting": "configured"}

    result = _resolve_placeholders(spec, config)

    assert result["inputs"]["trigger_value"] == "${trigger.data.value}"
    assert result["inputs"]["node_value"] == "${node1.output.result}"
    assert result["inputs"]["config_value"] == "configured"


@pytest.mark.unit
def test_resolve_placeholders_numeric_values():
    """Test placeholder resolution with numeric values."""
    spec = {
        "inputs": {
            "count": "${config.count}",
            "threshold": "${config.threshold}",
        }
    }

    config = {"count": 42, "threshold": 3.14}

    result = _resolve_placeholders(spec, config)

    assert result["inputs"]["count"] == "42"
    assert result["inputs"]["threshold"] == "3.14"


# =============================================================================
# Provider Connection Application Tests
# =============================================================================


@pytest.mark.unit
def test_apply_provider_connections():
    """Test applying provider connections to triggers."""
    spec = {
        "triggers": [
            {
                "id": "t1",
                "trigger": "gmail.new_email",
                "provider_config": {},
            },
            {
                "id": "t2",
                "trigger": "slack.message",
                "provider_config": {"existing": "value"},
            },
        ]
    }

    connections = {
        "gmail": 123,
        "slack": 456,
    }

    result = _apply_provider_connections(spec, connections)

    assert result["triggers"][0]["provider_config"]["connection_id"] == 123
    assert result["triggers"][1]["provider_config"]["connection_id"] == 456
    assert result["triggers"][1]["provider_config"]["existing"] == "value"


@pytest.mark.unit
def test_apply_provider_connections_no_triggers():
    """Test that spec without triggers is unchanged."""
    spec = {"nodes": [{"id": "n1"}]}
    connections = {"gmail": 123}

    result = _apply_provider_connections(spec, connections)

    assert result == spec


@pytest.mark.unit
def test_apply_provider_connections_partial():
    """Test applying connections when only some are provided."""
    spec = {
        "triggers": [
            {"id": "t1", "trigger": "gmail.new_email"},
            {"id": "t2", "trigger": "github.push"},
        ]
    }

    connections = {"gmail": 123}  # Only gmail provided

    result = _apply_provider_connections(spec, connections)

    assert result["triggers"][0]["provider_config"]["connection_id"] == 123
    assert "provider_config" not in result["triggers"][1]


# =============================================================================
# Integration Tests
# NOTE: Complex async database mocking tests are better covered by integration
# tests with actual database. The tests below verify the core logic works with
# mocked inputs. For full async database flow tests, see integration/templates/
# =============================================================================


@pytest.mark.unit
def test_to_summary_helper():
    """Test _to_summary helper function converts template correctly."""
    from seer.api.templates.services import _to_summary
    from seer.database import TemplateCategory

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.name = "Test Template"
    mock_template.description = "A test template"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.tags = ["test", "email"]
    mock_template.icon = "mail"
    mock_template.is_featured = True
    mock_template.usage_count = 42
    mock_template.required_integrations = [
        {"provider": "google", "integration_type": "gmail", "reason": "Send emails"}
    ]
    mock_template.visibility = "public"

    summary = _to_summary(mock_template)

    assert summary.template_id == "tpl_1"
    assert summary.slug == "test-template"
    assert summary.name == "Test Template"
    assert summary.category == "marketing"
    assert summary.tags == ["test", "email"]
    assert summary.is_featured is True
    assert summary.usage_count == 42
    assert len(summary.required_integrations) == 1
    assert summary.required_integrations[0].provider == "google"


@pytest.mark.unit
def test_to_detail_helper():
    """Test _to_detail helper function includes config fields."""
    from seer.api.templates.services import _to_detail
    from seer.database import TemplateCategory, TemplateSource
    from datetime import datetime, timezone

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.name = "Test Template"
    mock_template.description = "A test template"
    mock_template.category = TemplateCategory.MARKETING
    mock_template.source = TemplateSource.SYSTEM
    mock_template.tags = []
    mock_template.icon = None
    mock_template.preview_image_url = None
    mock_template.is_featured = False
    mock_template.usage_count = 0
    mock_template.required_integrations = []
    mock_template.spec = {
        "nodes": [
            {"inputs": {"to": "${config.recipient_email}"}}
        ]
    }
    mock_template.visibility = "public"
    mock_template.created_at = datetime.now(timezone.utc)
    mock_template.updated_at = datetime.now(timezone.utc)

    detail = _to_detail(mock_template)

    assert detail.slug == "test-template"
    assert len(detail.config_fields) == 1
    assert detail.config_fields[0].name == "recipient_email"
    assert detail.source == "system"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_check_requirements_builds_status():
    """Test check_requirements builds integration status correctly."""
    from seer.api.templates.services import check_requirements
    from seer.database import User

    user = MagicMock(spec=User)
    user.id = 1

    mock_template = MagicMock()
    mock_template.id = 1
    mock_template.slug = "test-template"
    mock_template.required_integrations = [
        {"provider": "google", "integration_type": "gmail", "reason": "Send emails"},
        {"provider": "slack", "integration_type": "slack", "reason": "Post messages"},
    ]

    # User only has Google connected
    mock_connection = MagicMock()
    mock_connection.id = 123
    mock_connection.provider = "google"
    mock_connection.provider_account_id = "user@gmail.com"

    with patch("seer.api.templates.services.WorkflowTemplate") as MockTemplate, \
         patch("seer.api.templates.services.OAuthConnection") as MockConnection:

        MockTemplate.filter.return_value.first = AsyncMock(return_value=mock_template)
        MockConnection.filter.return_value.all = AsyncMock(return_value=[mock_connection])

        result = await check_requirements(user, "test-template")

        assert result.all_connected is False
        assert len(result.integrations) == 2

        # Check Google is connected
        google_status = next(i for i in result.integrations if i.provider == "google")
        assert google_status.connected is True
        assert google_status.connection_id == 123

        # Check Slack is not connected
        slack_status = next(i for i in result.integrations if i.provider == "slack")
        assert slack_status.connected is False
        assert slack_status.connection_id is None
