"""
Unit tests for reference validation (Stage 3 of compiler).

Tests validation of ${...} references against type environment.
Target coverage: 90%+
"""
# pylint: disable=too-many-lines  # Comprehensive test coverage requires extensive test cases
import pytest

from seer.core.compiler.validate_refs import (
    validate_references,
    _uses_trigger_references,
    _node_uses_trigger_ids,
)
from seer.core.errors import ValidationPhaseError
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.schema.models import (
    Edge,
    EdgeType,
    ForEachNode,
    IfNode,
    ToolNode,
    TriggerSpec,
    WorkflowSpec,
)


# =============================================================================
# Valid Reference Tests
# =============================================================================


def test_validate_references_minimal_workflow():
    """Test validation passes for minimal workflow with no references."""
    spec = WorkflowSpec(version="2", triggers=[], nodes=[], edges=[])
    type_env = TypeEnvironment()

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_with_valid_trigger_ref():
    """Test validation passes for valid trigger reference."""
    type_env = TypeEnvironment()
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "${t1.data}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_with_node_output_ref():
    """Test validation passes for references to other node outputs."""
    type_env = TypeEnvironment()
    type_env.register("task1", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "hello"}
            ),
            ToolNode(
                id="task2",
                type="tool",
                tool="test.tool",
                inputs={"value": "${task1}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_with_multiple_refs():
    """Test validation with multiple valid references."""
    type_env = TypeEnvironment()
    type_env.register("t1", {"type": "object", "properties": {"x": {"type": "number"}}})
    type_env.register("t1.x", {"type": "number"})
    type_env.register("task1", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="trigger1",
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${t1.x}"}
            ),
            ToolNode(
                id="task2",
                type="tool",
                tool="test.tool",
                inputs={"value": "${task1}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger),
            Edge(source="task1", target="task2", type=EdgeType.default)
        ]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


# =============================================================================
# Invalid Reference Tests
# =============================================================================


def test_validate_references_missing_trigger_declaration():
    """Test error when trigger references are used but no triggers declared."""
    type_env = TypeEnvironment()
    # MyTrigger is registered but not in workflow triggers - should fail as undefined reference

    spec = WorkflowSpec(
        version="2",
        triggers=[],  # No triggers declared
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${MyTrigger.data}"}
            )
        ],
        edges=[]
    )

    # Should fail because MyTrigger is not defined in type environment
    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


def test_validate_references_undefined_symbol():
    """Test error when referencing undefined symbol."""
    type_env = TypeEnvironment()

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${undefined_var}"}
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


def test_validate_references_undefined_property():
    """Test error when referencing undefined property."""
    type_env = TypeEnvironment()
    type_env.register("task1", {"type": "object", "properties": {"name": {"type": "string"}}})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${task1.undefined_prop}"}
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


# =============================================================================
# Condition Node Validation Tests
# =============================================================================


def test_validate_references_if_node_condition():
    """Test validation of condition in IfNode."""
    type_env = TypeEnvironment()
    type_env.register("value", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            IfNode(
                id="if1",
                condition="${value} > 10"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_if_node_invalid_condition():
    """Test error when IfNode condition references undefined symbol."""
    type_env = TypeEnvironment()

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            IfNode(
                id="if1",
                condition="${undefined} > 10"
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


# =============================================================================
# ForEach Node Validation Tests
# =============================================================================


def test_validate_references_foreach_valid_items():
    """Test validation of ForEach node with valid items expression."""
    type_env = TypeEnvironment()
    type_env.register("items_list", {"type": "array", "items": {"type": "object"}})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                items="${items_list}",
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_foreach_non_array_items():
    """Test error when ForEach items expression is not an array."""
    type_env = TypeEnvironment()
    type_env.register("not_array", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                items="${not_array}",
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError, match="items expression must resolve to an array schema"):
        validate_references(spec, type_env)


def test_validate_references_foreach_undefined_items():
    """Test error when ForEach items expression references undefined symbol."""
    type_env = TypeEnvironment()

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                items="${undefined_items}",
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


def test_validate_references_foreach_complex_expression():
    """Test error when ForEach items is not a bare reference."""
    type_env = TypeEnvironment()
    type_env.register("items", {"type": "array"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                items="prefix_${items}",  # Not a bare reference
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError, match="Expression must be a bare"):
        validate_references(spec, type_env)


# =============================================================================
# Template String Validation Tests
# =============================================================================


def test_validate_references_template_string():
    """Test validation of template strings with multiple references."""
    type_env = TypeEnvironment()
    type_env.register("first", {"type": "string"})
    type_env.register("last", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "Hello ${first} ${last}!"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_template_string_invalid_ref():
    """Test error when template string contains invalid reference."""
    type_env = TypeEnvironment()
    type_env.register("first", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "Hello ${first} ${undefined}!"}
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError):
        validate_references(spec, type_env)


# =============================================================================
# Trigger Reference Detection Tests
# =============================================================================


def test_uses_trigger_references_true():
    """Test detection of trigger references in workflow."""
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "${t1.data}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    assert _uses_trigger_references(spec) is True


def test_uses_trigger_references_false():
    """Test that workflow without trigger references returns False."""
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "static value"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    assert _uses_trigger_references(spec) is False


def test_uses_trigger_references_no_triggers():
    """Test that workflow with no triggers returns False."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${other_var}"}
            )
        ],
        edges=[]
    )

    assert _uses_trigger_references(spec) is False


def test_node_uses_trigger_ids_task_node():
    """Test detection of trigger IDs in ToolNode."""
    node = ToolNode(
        id="task1",
        type="tool", tool="test.tool",
        inputs={"value": "${t1.data}"}
    )
    trigger_ids = {"t1"}

    assert _node_uses_trigger_ids(node, trigger_ids) is True


def test_node_uses_trigger_ids_if_node():
    """Test detection of trigger IDs in IfNode."""
    node = IfNode(
        id="if1",
        condition="${t1.value} > 10"
    )
    trigger_ids = {"t1"}

    assert _node_uses_trigger_ids(node, trigger_ids) is True


def test_node_uses_trigger_ids_foreach_node():
    """Test detection of trigger IDs in ForEachNode."""
    node = ForEachNode(
        id="loop1",
        items="${t1.items}",
        item_var="item",
        index_var="index"
    )
    trigger_ids = {"t1"}

    assert _node_uses_trigger_ids(node, trigger_ids) is True


def test_node_uses_trigger_ids_false():
    """Test that node without trigger references returns False."""
    node = ToolNode(
        id="task1",
        type="tool", tool="test.tool",
        inputs={"value": "${other_var}"}
    )
    trigger_ids = {"t1"}

    assert _node_uses_trigger_ids(node, trigger_ids) is False


# =============================================================================
# Multiple Error Collection Tests
# =============================================================================


def test_validate_references_collects_multiple_errors():
    """Test that validation collects and reports multiple errors."""
    type_env = TypeEnvironment()

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${undefined1}"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${undefined2}"}
            )
        ],
        edges=[]
    )

    with pytest.raises(ValidationPhaseError) as exc_info:
        validate_references(spec, type_env)

    # Error message should contain information about multiple errors
    error_msg = str(exc_info.value)
    assert "task1" in error_msg or "task2" in error_msg


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_validate_references_empty_value():
    """Test validation with empty string value (no references)."""
    type_env = TypeEnvironment()

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": ""}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_nested_object_access():
    """Test validation with deeply nested property access."""
    type_env = TypeEnvironment()
    type_env.register("data", {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"}
                        }
                    }
                }
            }
        }
    })
    type_env.register("data.level1", {"type": "object"})
    type_env.register("data.level1.level2", {"type": "object"})
    type_env.register("data.level1.level2.value", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${data.level1.level2.value}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


# =============================================================================
# Whitespace and Special Character Tests for 'out' Keys
# =============================================================================


def test_validate_references_out_key_with_spaces():
    """Test that 'out' keys with spaces work correctly."""
    type_env = TypeEnvironment()
    type_env.register("my task", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "hello"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${my task}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors - 'out' keys can have spaces
    validate_references(spec, type_env)


def test_validate_references_out_key_with_hyphens():
    """Test that 'out' keys with hyphens work correctly."""
    type_env = TypeEnvironment()
    type_env.register("task-result", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": 42}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${task-result}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors - 'out' keys can have hyphens
    validate_references(spec, type_env)


def test_validate_references_out_key_with_special_chars():
    """
    Test that 'out' keys with special characters work correctly.

    Note: Some characters like '@', '#', etc. can be used in 'out' keys and referenced,
    but '.', '[', ']' have special meaning in references (property/array access) so
    they can be in 'out' keys but cannot be referenced directly.
    """
    type_env = TypeEnvironment()
    type_env.register("task@result", {"type": "string"})
    type_env.register("data#point", {"type": "number"})
    type_env.register("result$value", {"type": "boolean"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${task@result}"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${data#point}"}
            ),
            ToolNode(
                id="task3",
                type="tool", tool="test.tool",
                inputs={"value": "${result$value}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors - 'out' keys can have special characters
    validate_references(spec, type_env)


def test_validate_references_out_key_with_unicode():
    """Test that 'out' keys with Unicode characters work correctly."""
    type_env = TypeEnvironment()
    type_env.register("résultat", {"type": "string"})
    type_env.register("数据", {"type": "object"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${résultat}"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${数据}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors - 'out' keys can have Unicode characters
    validate_references(spec, type_env)


def test_validate_references_out_key_with_nested_property_access():
    """Test that 'out' keys with spaces can have nested property access."""
    type_env = TypeEnvironment()
    type_env.register("my task", {
        "type": "object",
        "properties": {
            "result": {"type": "string"},
            "count": {"type": "number"}
        }
    })
    type_env.register("my task.result", {"type": "string"})
    type_env.register("my task.count", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${my task.result}"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${my task.count}"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_template_string_with_special_out_keys():
    """Test template strings with references to 'out' keys containing special characters."""
    type_env = TypeEnvironment()
    type_env.register("first-name", {"type": "string"})
    type_env.register("last name", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "Hello ${first-name} ${last name}!"}
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_if_node_with_special_out_keys():
    """Test IfNode condition with references to 'out' keys containing special characters."""
    type_env = TypeEnvironment()
    type_env.register("user-count", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            IfNode(
                id="if1",
                condition="${user-count} > 100"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_foreach_with_special_out_keys():
    """Test ForEachNode with references to 'out' keys containing special characters."""
    type_env = TypeEnvironment()
    type_env.register("data-items", {"type": "array", "items": {"type": "object"}})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                items="${data-items}",
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


# =============================================================================
# Whitespace and Special Character Tests for Trigger IDs
# =============================================================================


def test_validate_references_trigger_title_with_spaces():
    """Test that trigger IDs are used in references (not titles with spaces)."""
    type_env = TypeEnvironment()
    # Use trigger ID in references, not title
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="My Trigger",  # Title can have spaces, but references use ID
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "${t1.data}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    # Reference validation uses trigger ID, not title
    validate_references(spec, type_env)


def test_validate_references_trigger_title_with_hyphen():
    """Test that trigger IDs are used in references (titles are for display only)."""
    type_env = TypeEnvironment()
    type_env.register("t1", {"type": "object", "properties": {"value": {"type": "number"}}})
    type_env.register("t1.value", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="my-trigger",  # Title for display, ID for references
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${t1.value}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    # Reference validation uses trigger ID
    validate_references(spec, type_env)


def test_validate_references_trigger_title_unicode():
    """Test that trigger IDs are used in references even with Unicode titles."""
    type_env = TypeEnvironment()
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="触发器",  # Unicode title for display
                mode="polling",
                event_schema={},
            )
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "${t1.data}"}
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    # Reference validation uses trigger ID
    validate_references(spec, type_env)


def test_validate_references_complex_scenario_mixed_special_chars():
    """Test complex scenario with multiple nodes using various special characters."""
    type_env = TypeEnvironment()
    type_env.register("api-response", {"type": "object", "properties": {"status": {"type": "string"}}})
    type_env.register("api-response.status", {"type": "string"})
    type_env.register("user count", {"type": "number"})
    type_env.register("data@timestamp", {"type": "string"})
    type_env.register("result_array", {"type": "array", "items": {"type": "object"}})

    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${api-response.status}"}
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "Count: ${user count}, Time: ${data@timestamp}"}
            ),
            IfNode(
                id="if1",
                condition="${user count} > 0"
            ),
            ForEachNode(
                id="loop1",
                items="${result_array}",
                item_var="item",
                index_var="idx"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


# =============================================================================
# Multi-Trigger Validation Tests (Bug Fix: Reject bare "trigger" in multi-trigger)
# =============================================================================


def test_multi_trigger_rejects_bare_trigger_reference():
    """Test that multi-trigger workflows reject ${trigger.X} references."""
    type_env = TypeEnvironment()
    # Only register explicit trigger IDs, not "trigger"
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})
    type_env.register("t2", {"type": "object", "properties": {"payload": {"type": "string"}}})
    type_env.register("t2.payload", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(id="t1", key="trigger1.key", mode="polling", event_schema={}),
            TriggerSpec(id="t2", key="trigger2.key", mode="webhook", event_schema={}),
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${trigger.data}"}  # Invalid: bare "trigger" in multi-trigger workflow
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger),
            Edge(source="t2", target="task1", type=EdgeType.trigger),
        ]
    )

    # Should raise ValidationPhaseError with helpful message
    with pytest.raises(ValidationPhaseError) as exc_info:
        validate_references(spec, type_env)

    error_msg = str(exc_info.value)
    assert "Cannot use ${trigger.X} syntax in multi-trigger workflow" in error_msg
    assert "${t1.X}" in error_msg or "${t2.X}" in error_msg


def test_multi_trigger_accepts_explicit_trigger_ids():
    """Test that multi-trigger workflows accept explicit ${trigger_id.X} references."""
    type_env = TypeEnvironment()
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})
    type_env.register("t2", {"type": "object", "properties": {"payload": {"type": "string"}}})
    type_env.register("t2.payload", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(id="t1", key="trigger1.key", mode="polling", event_schema={}),
            TriggerSpec(id="t2", key="trigger2.key", mode="webhook", event_schema={}),
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool",
                tool="test.tool",
                inputs={"value": "${t1.data}"}  # Valid: explicit trigger ID
            ),
            ToolNode(
                id="task2",
                type="tool", tool="test.tool",
                inputs={"value": "${t2.payload}"}  # Valid: explicit trigger ID
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger),
            Edge(source="t2", target="task2", type=EdgeType.trigger),
        ]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_single_trigger_rejects_bare_trigger_reference():
    """Test that single-trigger workflows reject ${trigger.X} references."""
    type_env = TypeEnvironment()
    # Register only explicit ID (no "trigger" alias)
    type_env.register("t1", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("t1.data", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(id="t1", key="test.trigger", mode="polling", event_schema={}),
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${trigger.data}"}  # Invalid: bare "trigger" not allowed
            )
        ],
        edges=[
            Edge(source="t1", target="task1", type=EdgeType.trigger)
        ]
    )

    # Should raise ValidationPhaseError with helpful message
    with pytest.raises(ValidationPhaseError) as exc_info:
        validate_references(spec, type_env)

    error_msg = str(exc_info.value)
    assert "Cannot use ${trigger.X} syntax" in error_msg
    assert "${t1.X}" in error_msg


def test_single_trigger_accepts_explicit_id():
    """Test that single-trigger workflows accept ${trigger_id.X} references."""
    type_env = TypeEnvironment()
    type_env.register("email_trigger", {"type": "object", "properties": {"data": {"type": "object"}}})
    type_env.register("email_trigger.data", {"type": "object"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(id="email_trigger", key="poll.gmail.email_received", mode="polling", event_schema={}),
        ],
        nodes=[
            ToolNode(
                id="task1",
                type="tool", tool="test.tool",
                inputs={"value": "${email_trigger.data}"}  # Valid: explicit ID
            )
        ],
        edges=[
            Edge(source="email_trigger", target="task1", type=EdgeType.trigger)
        ]
    )

    # Should not raise any errors
    validate_references(spec, type_env)
