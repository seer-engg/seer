"""
Unit tests for type environment building (Stage 2 of compiler).

Tests type environment construction, symbol registration, and schema inference.
Target coverage: 90%+
"""
import pytest

from seer.core.compiler.type_env import (
    VALID_IDENTIFIER,
    build_type_environment,
    _infer_schema_from_value,
    _register_triggers,
)
from seer.core.errors import TypeEnvironmentError
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    ForEachNode,
    LLMNode,
    OutputContract,
    OutputMode,
    TaskKind,
    TaskNode,
    ToolNode,
    TriggerSpec,
    TriggerSchemas,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry


# =============================================================================
# Valid Identifier Tests
# =============================================================================


@pytest.mark.parametrize("valid_identifier", [
    "trigger1",
    "MyTrigger",
    "_trigger",
    "Gmail_Inbox",
    "webhook_1",
    "a",
    "_",
    "CamelCase123",
])
def test_valid_identifier_pattern(valid_identifier):
    """Test that valid identifiers match the VALID_IDENTIFIER pattern."""
    assert VALID_IDENTIFIER.match(valid_identifier) is not None


@pytest.mark.parametrize("invalid_identifier", [
    "1trigger",  # starts with number
    "trigger-name",  # contains hyphen
    "trigger.name",  # contains dot
    "trigger name",  # contains space
    "trigger@email",  # contains special char
    "",  # empty string
])
def test_invalid_identifier_pattern(invalid_identifier):
    """Test that invalid identifiers do not match the VALID_IDENTIFIER pattern."""
    assert VALID_IDENTIFIER.match(invalid_identifier) is None


# =============================================================================
# Trigger Registration Tests
# =============================================================================


def test_register_triggers_basic():
    """Test basic trigger registration."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="TestTrigger",
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(
                event={"type": "object", "properties": {"message": {"type": "string"}}}
            )
        )
    ]

    _register_triggers(triggers, env)

    # Trigger title should be registered
    symbols = env.as_dict()
    assert "TestTrigger" in symbols
    # Sub-properties should be registered
    assert "TestTrigger.message" in symbols


def test_register_triggers_with_default_schema():
    """Test trigger registration with default event schema."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="SimpleTrigger",
            provider="test",
            mode="polling",
            schemas=TriggerSchemas()  # Use default (empty dict) for event schema
        )
    ]

    _register_triggers(triggers, env)

    # Should use default schema with additionalProperties: True
    symbols = env.as_dict()
    assert "SimpleTrigger" in symbols
    schema = symbols["SimpleTrigger"]
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is True


def test_register_triggers_duplicate_title_error():
    """Test that duplicate trigger titles raise TypeEnvironmentError."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="trigger1.key",
            title="DuplicateTitle",
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        ),
        TriggerSpec(
            id="t2",
            key="trigger2.key",
            title="DuplicateTitle",  # Duplicate!
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    with pytest.raises(TypeEnvironmentError, match="Duplicate trigger title 'DuplicateTitle'"):
        _register_triggers(triggers, env)


def test_register_triggers_with_hyphen_in_title():
    """Test that trigger titles with hyphens are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="invalid-title",  # Valid: hyphens are now allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "invalid-title" in symbols


@pytest.mark.parametrize("valid_title", [
    "1StartWithNumber",
    "has-hyphen",
    "has.dot",
    "has space",
])
def test_register_triggers_various_special_char_titles(valid_title):
    """Test that trigger titles can now contain various special characters."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title=valid_title,  # All special chars are now valid
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert valid_title in symbols


# =============================================================================
# Schema Inference Tests
# =============================================================================


def test_infer_schema_from_string():
    """Test schema inference for string values."""
    schema = _infer_schema_from_value("hello")
    assert schema == {"type": "string"}


def test_infer_schema_from_integer():
    """Test schema inference for integer values."""
    schema = _infer_schema_from_value(42)
    assert schema == {"type": "integer"}


def test_infer_schema_from_float():
    """Test schema inference for float values."""
    schema = _infer_schema_from_value(3.14)
    assert schema == {"type": "number"}


def test_infer_schema_from_boolean():
    """Test schema inference for boolean values."""
    schema = _infer_schema_from_value(True)
    assert schema == {"type": "boolean"}


def test_infer_schema_from_null():
    """Test schema inference for null values."""
    schema = _infer_schema_from_value(None)
    assert schema == {"type": "null"}


def test_infer_schema_from_empty_list():
    """Test schema inference for empty list."""
    schema = _infer_schema_from_value([])
    assert schema == {"type": "array"}


def test_infer_schema_from_list_with_items():
    """Test schema inference for list with items."""
    schema = _infer_schema_from_value([1, 2, 3])
    assert schema == {"type": "array", "items": {"type": "integer"}}


def test_infer_schema_from_nested_list():
    """Test schema inference for nested list."""
    schema = _infer_schema_from_value([["nested"]])
    assert schema["type"] == "array"
    assert schema["items"]["type"] == "array"
    assert schema["items"]["items"] == {"type": "string"}


def test_infer_schema_from_empty_object():
    """Test schema inference for empty object."""
    schema = _infer_schema_from_value({})
    assert schema["type"] == "object"
    assert schema["properties"] == {}
    assert schema.get("additionalProperties") is True


def test_infer_schema_from_object_with_properties():
    """Test schema inference for object with properties."""
    schema = _infer_schema_from_value({"name": "John", "age": 30})
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["age"] == {"type": "integer"}
    assert schema.get("additionalProperties") is True


def test_infer_schema_from_nested_object():
    """Test schema inference for nested object."""
    schema = _infer_schema_from_value({
        "user": {
            "name": "John",
            "contact": {
                "email": "john@example.com"
            }
        }
    })
    assert schema["type"] == "object"
    assert schema["properties"]["user"]["type"] == "object"
    assert schema["properties"]["user"]["properties"]["contact"]["type"] == "object"


def test_infer_schema_from_complex_structure():
    """Test schema inference for complex nested structure."""
    value = {
        "items": [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"}
        ],
        "total": 2,
        "active": True
    }

    schema = _infer_schema_from_value(value)
    assert schema["type"] == "object"
    assert schema["properties"]["items"]["type"] == "array"
    assert schema["properties"]["items"]["items"]["type"] == "object"
    assert schema["properties"]["total"] == {"type": "integer"}
    assert schema["properties"]["active"] == {"type": "boolean"}


# =============================================================================
# Build Type Environment Tests
# =============================================================================


def test_build_type_environment_minimal():
    """Test building type environment for minimal workflow."""
    spec = WorkflowSpec(version="2", triggers=[], nodes=[], edges=[])
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    assert isinstance(env, TypeEnvironment)
    assert len(env.as_dict()) == 0


def test_build_type_environment_with_trigger():
    """Test building type environment with trigger."""
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="t1",
                key="test.trigger",
                title="MyTrigger",
                provider="test",
                mode="polling",
                schemas=TriggerSchemas(
                    event={"type": "object", "properties": {"data": {"type": "string"}}}
                )
            )
        ],
        nodes=[],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "MyTrigger" in symbols
    assert "MyTrigger.data" in symbols


def test_build_type_environment_with_task_node():
    """Test building type environment with task node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value="hello world",
                out="result"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "result" in symbols
    assert symbols["result"]["type"] == "string"


def test_build_type_environment_with_foreach_node():
    """Test building type environment with ForEach node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                type="for_each",
                items="${items}",
                item_var="item",
                index_var="index",
                out="results"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    # Loop variables should be registered
    symbols = env.as_dict()
    assert "item" in symbols
    assert "index" in symbols
    assert symbols["index"]["type"] == "integer"

    # Output should be registered
    assert "results" in symbols


def test_build_type_environment_with_llm_node():
    """Test building type environment with LLM node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            LLMNode(
                id="llm1",
                type="llm",
                model="gpt-4",
                prompt="Generate a response",
                output=OutputContract(mode=OutputMode.json, schema={"schema": {"type": "object"}}),
                out="llm_result"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "llm_result" in symbols


def test_build_type_environment_with_multiple_nodes():
    """Test building type environment with multiple nodes."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value=42,
                out="number"
            ),
            TaskNode(
                id="task2",
                type="task",
                kind=TaskKind.set,
                value="text",
                out="text"
            ),
            TaskNode(
                id="task3",
                type="task",
                kind=TaskKind.set,
                value=True,
                out="flag"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "number" in symbols
    assert "text" in symbols
    assert "flag" in symbols
    assert symbols["number"]["type"] == "integer"
    assert symbols["text"]["type"] == "string"
    assert symbols["flag"]["type"] == "boolean"


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_build_type_environment_node_without_output():
    """Test building type environment with node that has no output."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value="hello"
                # No 'out' field
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    # No symbol should be registered since there's no output
    assert len(env.as_dict()) == 0


def test_build_type_environment_foreach_without_output():
    """Test ForEach node without output still registers loop variables."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                type="for_each",
                items="${items}",
                item_var="item",
                index_var="idx"
                # No 'out' field
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    # Loop variables should still be registered
    symbols = env.as_dict()
    assert "item" in symbols
    assert "idx" in symbols


# =============================================================================
# Special Character Tests for Trigger Titles
# =============================================================================


def test_register_triggers_title_with_spaces():
    """Test that trigger titles with spaces are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="My Trigger",  # Valid: spaces are now allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "My Trigger" in symbols


def test_register_triggers_title_with_multiple_spaces():
    """Test that trigger titles with multiple spaces are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="My  Complex  Trigger  Name",  # Valid: multiple spaces allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "My  Complex  Trigger  Name" in symbols


def test_register_triggers_title_with_another_hyphen():
    """Test that trigger titles with hyphens are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="my-trigger",  # Valid: hyphens allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "my-trigger" in symbols


def test_register_triggers_title_with_dot():
    """
    Test that trigger titles with dots are accepted during registration.

    Note: While dots are allowed in trigger titles, they can create ambiguity
    when used in references because dots have special meaning for property access.
    For example, if a trigger title is "trigger.name" and you reference
    "${trigger.name.property}", it's ambiguous whether this means:
    - The trigger "trigger.name" with property "property", or
    - The trigger "trigger" with nested property "name.property"

    It's recommended to avoid dots in trigger titles to prevent confusion.
    """
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="trigger.name",  # Valid: dots allowed, but may cause ambiguity
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error during registration
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "trigger.name" in symbols


def test_register_triggers_title_with_at_sign():
    """Test that trigger titles with @ sign are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="trigger@email",  # Valid: @ allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "trigger@email" in symbols


def test_register_triggers_title_starts_with_number():
    """Test that trigger titles starting with a number are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="1trigger",  # Valid: can start with number
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "1trigger" in symbols


def test_register_triggers_title_with_unicode():
    """Test that trigger titles with Unicode characters are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title="触发器",  # Valid: Unicode allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert "触发器" in symbols


@pytest.mark.parametrize("special_char_title", [
    "trigger!name",
    "trigger#tag",
    "trigger$var",
    "trigger%percent",
    "trigger&and",
    "trigger*star",
    "trigger(paren",
    "trigger)paren",
    "trigger+plus",
    "trigger=equals",
    "trigger[bracket",
    "trigger]bracket",
    "trigger{brace",
    "trigger}brace",
    "trigger|pipe",
    "trigger\\backslash",
    "trigger:colon",
    "trigger;semicolon",
    "trigger'quote",
    'trigger"doublequote',
    "trigger<less",
    "trigger>greater",
    "trigger?question",
    "trigger/slash",
    "trigger~tilde",
    "trigger`backtick",
])
def test_register_triggers_title_with_various_special_chars(special_char_title):
    """Test that trigger titles with various special characters are now accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            title=special_char_title,  # Valid: all special chars allowed
            provider="test",
            mode="polling",
            schemas=TriggerSchemas(event={})
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger is registered
    symbols = env.as_dict()
    assert special_char_title in symbols


# =============================================================================
# Special Character Tests for Node 'out' Keys in Type Environment
# =============================================================================


def test_build_type_environment_out_key_with_spaces():
    """Test that 'out' keys with spaces are accepted during type environment building."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value="hello world",
                out="my result"  # Valid: out keys can have spaces
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "my result" in symbols
    assert symbols["my result"]["type"] == "string"


def test_build_type_environment_out_key_with_hyphens():
    """Test that 'out' keys with hyphens are accepted."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value=42,
                out="task-result"  # Valid: out keys can have hyphens
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "task-result" in symbols
    assert symbols["task-result"]["type"] == "integer"


def test_build_type_environment_out_key_with_special_chars():
    """Test that 'out' keys with various special characters are accepted."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value="test",
                out="data@2024"  # Valid: out keys can have @ character
            ),
            TaskNode(
                id="task2",
                type="task",
                kind=TaskKind.set,
                value=100,
                out="result#value"  # Valid: out keys can have # character
            ),
            TaskNode(
                id="task3",
                type="task",
                kind=TaskKind.set,
                value=True,
                out="flag$state"  # Valid: out keys can have $ character
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "data@2024" in symbols
    assert "result#value" in symbols
    assert "flag$state" in symbols


def test_build_type_environment_out_key_with_unicode():
    """Test that 'out' keys with Unicode characters are accepted."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            TaskNode(
                id="task1",
                type="task",
                kind=TaskKind.set,
                value="success",
                out="résultat"  # Valid: Unicode in out key
            ),
            TaskNode(
                id="task2",
                type="task",
                kind=TaskKind.set,
                value={"data": "value"},
                out="数据"  # Valid: Chinese characters
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "résultat" in symbols
    assert "数据" in symbols
