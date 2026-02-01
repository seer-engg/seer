"""
Unit tests for type environment building (Stage 2 of compiler).

Tests type environment construction, symbol registration, and schema inference.
Target coverage: 90%+
"""
# pylint: disable=too-many-lines  # Comprehensive test coverage requires extensive test cases
import pytest

from seer.core.compiler.type_env import (
    VALID_IDENTIFIER,
    build_type_environment,
    _register_triggers,
)
from seer.core.expr.typecheck import TypeEnvironment
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.models import (
    ForEachNode,
    LLMNode,
    OutputContract,
    OutputMode,
    TriggerSpec,
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
            mode="polling",
            event_schema={"type": "object", "properties": {"message": {"type": "string"}}},
        )
    ]

    _register_triggers(triggers, env)

    # Trigger ID should be registered
    symbols = env.as_dict()
    assert "t1" in symbols
    # Sub-properties should be registered
    assert "t1.message" in symbols


def test_register_triggers_with_default_schema():
    """Test trigger registration with default event schema."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="test.trigger",
            mode="polling",
            # Use default (empty dict) for event_schema
        )
    ]

    _register_triggers(triggers, env)

    # Should use default schema with additionalProperties: True
    symbols = env.as_dict()
    assert "t1" in symbols
    schema = symbols["t1"]
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is True


def test_register_triggers_duplicate_id_allowed_different_titles():
    """Test that triggers with duplicate titles but different IDs are allowed.

    Note: Duplicate trigger IDs are caught at WorkflowSpec validation level,
    not during type environment building. This test verifies that triggers
    with different IDs can have the same title without issues.
    """
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="t1",
            key="trigger1.key",
            mode="polling",
            event_schema={},
        ),
        TriggerSpec(
            id="t2",
            key="trigger2.key",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error since IDs are different
    _register_triggers(triggers, env)

    # Both IDs should be registered
    symbols = env.as_dict()
    assert "t1" in symbols
    assert "t2" in symbols


def test_register_triggers_with_hyphen_in_id():
    """Test that trigger IDs with hyphens are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="trigger-with-hyphen",
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "trigger-with-hyphen" in symbols


@pytest.mark.parametrize("valid_id", [
    "1StartWithNumber",
    "has-hyphen",
    "has.dot",
    "has space",
])
def test_register_triggers_various_special_char_ids(valid_id):
    """Test that trigger IDs can contain various special characters."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id=valid_id,  # All special chars are now valid in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert valid_id in symbols


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
                mode="polling",
                event_schema={"type": "object", "properties": {"data": {"type": "string"}}},
            )
        ],
        nodes=[],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "t1" in symbols
    assert "t1.data" in symbols


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
                index_var="index"
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

    # Node ID should be registered as output
    assert "loop1" in symbols


def test_build_type_environment_with_llm_node():
    """Test building type environment with LLM node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            LLMNode(
                id="llm1",
                type="llm",
                inputs={"model": "gpt-4", "prompt": "Generate a response"},
                outputs=OutputContract(mode=OutputMode.json, schema={"schema": {"type": "object"}})
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()
    assert "llm1" in symbols


# =============================================================================
# Edge Case Tests
# =============================================================================


def test_build_type_environment_node_with_output():
    """Test building type environment with node that produces output.

    Note: All nodes with values now register their ID as output in the type environment.
    """
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            LLMNode(
                id="llm1",
                type="llm",
                inputs={"model": "gpt-4", "prompt": "test"}
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    # Node ID should be registered as symbol
    symbols = env.as_dict()
    assert "llm1" in symbols
    assert symbols["llm1"]["type"] == "string"


def test_build_type_environment_foreach_without_output():
    """Test ForEach node without output contract still registers loop variables and node ID."""
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
                # No 'output' field
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
    assert "idx" in symbols

    # Node ID should be registered with default array schema
    assert "loop1" in symbols
    assert symbols["loop1"]["type"] == "array"


# =============================================================================
# Special Character Tests for Trigger IDs
# =============================================================================


def test_register_triggers_id_with_spaces():
    """Test that trigger IDs with spaces are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="My Trigger",  # Valid: spaces are allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "My Trigger" in symbols


def test_register_triggers_id_with_multiple_spaces():
    """Test that trigger IDs with multiple spaces are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="My  Complex  Trigger  ID",  # Valid: multiple spaces allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "My  Complex  Trigger  ID" in symbols


def test_register_triggers_id_with_another_hyphen():
    """Test that trigger IDs with hyphens are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="my-trigger-id",  # Valid: hyphens allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "my-trigger-id" in symbols


def test_register_triggers_id_with_dot():
    """
    Test that trigger IDs with dots are accepted during registration.

    Note: While dots are allowed in trigger IDs, they can create ambiguity
    when used in references because dots have special meaning for property access.
    For example, if a trigger ID is "trigger.name" and you reference
    "${trigger.name.property}", it's ambiguous whether this means:
    - The trigger "trigger.name" with property "property", or
    - The trigger "trigger" with nested property "name.property"

    It's recommended to avoid dots in trigger IDs to prevent confusion.
    """
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="trigger.id",  # Valid: dots allowed in IDs, but may cause ambiguity
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error during registration
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "trigger.id" in symbols


def test_register_triggers_id_with_at_sign():
    """Test that trigger IDs with @ sign are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="trigger@email",  # Valid: @ allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "trigger@email" in symbols


def test_register_triggers_id_starts_with_number():
    """Test that trigger IDs starting with a number are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="1trigger",  # Valid: IDs can start with number
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "1trigger" in symbols


def test_register_triggers_id_with_unicode():
    """Test that trigger IDs with Unicode characters are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="触发器",  # Valid: Unicode allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert "触发器" in symbols


@pytest.mark.parametrize("special_char_id", [
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
def test_register_triggers_id_with_various_special_chars(special_char_id):
    """Test that trigger IDs with various special characters are accepted."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id=special_char_id,  # Valid: all special chars allowed in IDs
            key="test.trigger",
            mode="polling",
            event_schema={},
        )
    ]

    # Should not raise an error
    _register_triggers(triggers, env)

    # Verify trigger ID is registered
    symbols = env.as_dict()
    assert special_char_id in symbols


# =============================================================================
# Single-Trigger Alias Tests (Bug Fix: Register "trigger" for single-trigger workflows)
# =============================================================================


def test_single_trigger_does_not_register_trigger_alias():
    """Test that single-trigger workflows do NOT register 'trigger' symbol."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="gmail_trigger",
            key="poll.gmail.email_received",
            mode="polling",
            event_schema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "message_id": {"type": "string"},
                            "from": {"type": "string"},
                        }
                    }
                }
            },
        )
    ]

    _register_triggers(triggers, env)

    symbols = env.as_dict()

    # Explicit ID should be registered
    assert "gmail_trigger" in symbols
    assert "gmail_trigger.data" in symbols

    # "trigger" should NOT be registered
    assert "trigger" not in symbols
    assert "trigger.data" not in symbols


def test_multi_trigger_does_not_register_trigger_alias():
    """Test that multi-trigger workflows do NOT register 'trigger' symbol."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="gmail_trigger",
            key="poll.gmail.email_received",
            mode="polling",
            event_schema={"type": "object", "properties": {"data": {"type": "object"}}},
        ),
        TriggerSpec(
            id="webhook_trigger",
            key="webhook.generic",
            mode="webhook",
            event_schema={"type": "object", "properties": {"payload": {"type": "object"}}},
        ),
    ]

    _register_triggers(triggers, env)

    symbols = env.as_dict()

    # Explicit IDs should be registered
    assert "gmail_trigger" in symbols
    assert "webhook_trigger" in symbols

    # Multi-trigger: "trigger" should NOT be registered
    assert "trigger" not in symbols
    assert "trigger.data" not in symbols


def test_single_trigger_with_no_properties():
    """Test single-trigger registration with event schema having no properties."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="simple_trigger",
            key="test.trigger",
            mode="polling",
            event_schema={"type": "object"},  # No properties field
        )
    ]

    _register_triggers(triggers, env)

    symbols = env.as_dict()

    # Only explicit ID should be registered
    assert "simple_trigger" in symbols
    assert "trigger" not in symbols

    # No sub-properties should be registered (since no properties defined)
    assert "simple_trigger.data" not in symbols
    assert "trigger.data" not in symbols


def test_single_trigger_with_nested_properties():
    """Test single-trigger registration extracts all top-level properties."""
    env = TypeEnvironment()
    triggers = [
        TriggerSpec(
            id="email_trigger",
            key="poll.gmail.email_received",
            mode="polling",
            event_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object"},
                    "id": {"type": "string"},
                    "timestamp": {"type": "string"},
                }
            },
        )
    ]

    _register_triggers(triggers, env)

    symbols = env.as_dict()

    # Verify all properties registered for trigger ID only
    assert "email_trigger" in symbols
    assert "email_trigger.data" in symbols
    assert "email_trigger.id" in symbols
    assert "email_trigger.timestamp" in symbols

    # "trigger" alias should NOT be registered
    assert "trigger" not in symbols
    assert "trigger.data" not in symbols
