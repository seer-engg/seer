"""
Unit tests for reference validation (Stage 3 of compiler).

Tests validation of ${...} references against type environment.
Target coverage: 90%+
"""
import pytest

from seer.core.compiler.validate_refs import (
    validate_references,
    _uses_trigger_references,
    _node_uses_trigger_titles,
)
from seer.core.errors import ValidationPhaseError
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.schema.models import (
    ForEachNode,
    IfNode,
    TaskKind,
    TaskNode,
    TriggerSpec,
    TriggerSchemas,
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
    type_env.register("MyTrigger", {"type": "object", "properties": {"data": {"type": "string"}}})
    type_env.register("MyTrigger.data", {"type": "string"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="MyTrigger",
                provider="test",
                mode="polling",
                schemas=TriggerSchemas(event={})
            )
        ],
        nodes=[
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${MyTrigger.data}",
                out="result"
            )
        ],
        edges=[]
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="hello",
                out="task1"
            ),
            TaskNode(
                id="task2",
                kind=TaskKind.set,
                value="${task1}",
                out="task2"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)


def test_validate_references_with_multiple_refs():
    """Test validation with multiple valid references."""
    type_env = TypeEnvironment()
    type_env.register("trigger1", {"type": "object", "properties": {"x": {"type": "number"}}})
    type_env.register("trigger1.x", {"type": "number"})
    type_env.register("task1", {"type": "number"})

    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="trigger1",
                provider="test",
                mode="polling",
                schemas=TriggerSchemas(event={})
            )
        ],
        nodes=[
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${trigger1.x}",
                out="task1"
            ),
            TaskNode(
                id="task2",
                kind=TaskKind.set,
                value="${task1}",
                out="task2"
            )
        ],
        edges=[]
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${MyTrigger.data}",
                out="result"
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${undefined_var}",
                out="result"
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
            TaskNode(
                id="task2",
                kind=TaskKind.set,
                value="${task1.undefined_prop}",
                out="result"
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="Hello ${first} ${last}!",
                out="greeting"
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="Hello ${first} ${undefined}!",
                out="greeting"
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
                title="MyTrigger",
                provider="test",
                mode="polling",
                schemas=TriggerSchemas(event={})
            )
        ],
        nodes=[
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${MyTrigger.data}",
                out="result"
            )
        ],
        edges=[]
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
                title="MyTrigger",
                provider="test",
                mode="polling",
                schemas=TriggerSchemas(event={})
            )
        ],
        nodes=[
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="static value",
                out="result"
            )
        ],
        edges=[]
    )

    assert _uses_trigger_references(spec) is False


def test_uses_trigger_references_no_triggers():
    """Test that workflow with no triggers returns False."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${other_var}",
                out="result"
            )
        ],
        edges=[]
    )

    assert _uses_trigger_references(spec) is False


def test_node_uses_trigger_titles_task_node():
    """Test detection of trigger titles in TaskNode."""
    node = TaskNode(
        id="task1",
        kind=TaskKind.set,
        value="${TriggerTitle.data}",
        out="result"
    )
    trigger_titles = {"TriggerTitle"}

    assert _node_uses_trigger_titles(node, trigger_titles) is True


def test_node_uses_trigger_titles_if_node():
    """Test detection of trigger titles in IfNode."""
    node = IfNode(
        id="if1",
        condition="${TriggerTitle.value} > 10"
    )
    trigger_titles = {"TriggerTitle"}

    assert _node_uses_trigger_titles(node, trigger_titles) is True


def test_node_uses_trigger_titles_foreach_node():
    """Test detection of trigger titles in ForEachNode."""
    node = ForEachNode(
        id="loop1",
        items="${TriggerTitle.items}",
        item_var="item",
        index_var="index"
    )
    trigger_titles = {"TriggerTitle"}

    assert _node_uses_trigger_titles(node, trigger_titles) is True


def test_node_uses_trigger_titles_false():
    """Test that node without trigger references returns False."""
    node = TaskNode(
        id="task1",
        kind=TaskKind.set,
        value="${other_var}",
        out="result"
    )
    trigger_titles = {"TriggerTitle"}

    assert _node_uses_trigger_titles(node, trigger_titles) is False


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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${undefined1}",
                out="result1"
            ),
            TaskNode(
                id="task2",
                kind=TaskKind.set,
                value="${undefined2}",
                out="result2"
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="",
                out="result"
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
            TaskNode(
                id="task1",
                kind=TaskKind.set,
                value="${data.level1.level2.value}",
                out="result"
            )
        ],
        edges=[]
    )

    # Should not raise any errors
    validate_references(spec, type_env)
