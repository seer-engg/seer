"""
Unit tests for workflow spec parsing (Stage 1 of compiler).

Tests JSON parsing, validation, and error handling for workflow specifications.
Target coverage: 90%+
"""
import json

import pytest

from seer.core.compiler.parse import parse_workflow_spec
from seer.core.errors import ValidationPhaseError
from seer.core.schema.models import WorkflowSpec


# =============================================================================
# Valid Workflow Parsing Tests
# =============================================================================


def test_parse_valid_workflow_from_dict(sample_workflow_spec):
    """Test parsing valid workflow spec from dictionary."""
    spec = parse_workflow_spec(sample_workflow_spec)

    assert isinstance(spec, WorkflowSpec)
    assert spec.version == "2"
    assert len(spec.triggers) == 1
    assert len(spec.nodes) == 1
    assert len(spec.edges) == 1
    assert spec.triggers[0].id == "t1"
    assert spec.nodes[0].id == "n1"


def test_parse_valid_workflow_from_json_string(sample_workflow_spec):
    """Test parsing valid workflow spec from JSON string."""
    json_string = json.dumps(sample_workflow_spec)
    spec = parse_workflow_spec(json_string)

    assert isinstance(spec, WorkflowSpec)
    assert spec.version == "2"
    assert len(spec.triggers) == 1


def test_parse_minimal_workflow():
    """Test parsing minimal valid workflow with no triggers."""
    minimal_spec = {
        "version": "2",
        "triggers": [],
        "nodes": [],
        "edges": []
    }

    spec = parse_workflow_spec(minimal_spec)

    assert isinstance(spec, WorkflowSpec)
    assert spec.version == "2"
    assert spec.triggers == []
    assert spec.nodes == []
    assert spec.edges == []


def test_parse_complex_workflow(complex_workflow_spec):
    """Test parsing complex workflow with multiple nodes and edges."""
    spec = parse_workflow_spec(complex_workflow_spec)

    assert isinstance(spec, WorkflowSpec)
    assert len(spec.nodes) == 4
    assert len(spec.edges) == 4

    # Verify node types
    node_types = {node.type for node in spec.nodes}
    assert "task" in node_types
    assert "condition" in node_types


# =============================================================================
# Invalid JSON Tests
# =============================================================================


def test_parse_invalid_json_string():
    """Test that invalid JSON string raises ValidationPhaseError."""
    invalid_json = "{invalid json: missing quotes}"

    with pytest.raises(ValidationPhaseError, match="Invalid workflow JSON payload"):
        parse_workflow_spec(invalid_json)


def test_parse_malformed_json_string():
    """Test that malformed JSON raises ValidationPhaseError."""
    malformed_json = '{"version": "2", "triggers": [,]}'

    with pytest.raises(ValidationPhaseError, match="Invalid workflow JSON payload"):
        parse_workflow_spec(malformed_json)


def test_parse_empty_json_string():
    """Test that empty JSON string raises ValidationPhaseError."""
    with pytest.raises(ValidationPhaseError, match="Invalid workflow JSON payload"):
        parse_workflow_spec("")


# =============================================================================
# Invalid Payload Type Tests
# =============================================================================


def test_parse_unsupported_payload_type_list():
    """Test that list payload raises ValidationPhaseError."""
    with pytest.raises(ValidationPhaseError, match="Unsupported payload type list"):
        parse_workflow_spec([1, 2, 3])


def test_parse_unsupported_payload_type_number():
    """Test that number payload raises ValidationPhaseError."""
    with pytest.raises(ValidationPhaseError, match="Unsupported payload type int"):
        parse_workflow_spec(123)


def test_parse_unsupported_payload_type_none():
    """Test that None payload raises ValidationPhaseError."""
    with pytest.raises(ValidationPhaseError, match="Unsupported payload type NoneType"):
        parse_workflow_spec(None)


# =============================================================================
# Invalid Spec Structure Tests
# =============================================================================


def test_parse_missing_version():
    """Test that spec without version raises ValidationPhaseError."""
    invalid_spec = {
        "triggers": [],
        "nodes": [],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_invalid_version():
    """Test that spec with invalid version raises ValidationPhaseError."""
    invalid_spec = {
        "version": "999",
        "triggers": [],
        "nodes": [],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_missing_triggers():
    """Test that spec without triggers field raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "nodes": [],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_missing_nodes():
    """Test that spec without nodes field raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "triggers": [],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_missing_edges():
    """Test that spec without edges field raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "triggers": [],
        "nodes": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_invalid_trigger_structure():
    """Test that invalid trigger structure raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                # Missing required 'key' field
                "label": "Test Trigger"
            }
        ],
        "nodes": [],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_invalid_node_structure():
    """Test that invalid node structure raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "n1",
                # Missing required 'type' field
                "label": "Test Node"
            }
        ],
        "edges": []
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


def test_parse_invalid_edge_structure():
    """Test that invalid edge structure raises ValidationPhaseError."""
    invalid_spec = {
        "version": "2",
        "triggers": [],
        "nodes": [],
        "edges": [
            {
                "id": "e1",
                # Missing required 'source' and 'target' fields
                "label": "Test Edge"
            }
        ]
    }

    with pytest.raises(ValidationPhaseError, match="Workflow spec validation failed"):
        parse_workflow_spec(invalid_spec)


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_parse_workflow_with_extra_fields():
    """Test that workflow spec with extra fields still parses successfully."""
    spec_with_extras = {
        "version": "2",
        "triggers": [],
        "nodes": [],
        "edges": [],
        "extra_field": "should be ignored",
        "another_extra": 123
    }

    spec = parse_workflow_spec(spec_with_extras)

    assert isinstance(spec, WorkflowSpec)
    assert spec.version == "2"


def test_parse_workflow_with_unicode_characters():
    """Test parsing workflow with unicode characters in labels."""
    unicode_spec = {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "label": "测试触发器 🚀",
                "config": {}
            }
        ],
        "nodes": [],
        "edges": []
    }

    spec = parse_workflow_spec(unicode_spec)

    assert isinstance(spec, WorkflowSpec)
    assert spec.triggers[0].label == "测试触发器 🚀"


def test_parse_deeply_nested_node_config():
    """Test parsing workflow with deeply nested node configuration."""
    nested_spec = {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "n1",
                "type": "task",
                "label": "Nested Task",
                "config": {
                    "tool_call": {
                        "tool_id": "test.tool",
                        "parameters": {
                            "nested": {
                                "deeply": {
                                    "very_deep": {
                                        "value": "found"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ],
        "edges": []
    }

    spec = parse_workflow_spec(nested_spec)

    assert isinstance(spec, WorkflowSpec)
    assert len(spec.nodes) == 1


# =============================================================================
# Parametrized Tests for Multiple Invalid Cases
# =============================================================================


@pytest.mark.parametrize("invalid_input,expected_error", [
    (True, "Unsupported payload type bool"),
    (False, "Unsupported payload type bool"),
    (3.14, "Unsupported payload type float"),
    ({"version": "1"}, "Workflow spec validation failed"),  # Invalid version
])
def test_parse_various_invalid_inputs(invalid_input, expected_error):
    """Test various invalid inputs raise appropriate errors."""
    with pytest.raises(ValidationPhaseError, match=expected_error):
        parse_workflow_spec(invalid_input)
