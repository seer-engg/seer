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
    AgentNode,
    ForEachNode,
    OutputContract,
    OutputMode,
    TriggerSpec,
    WorkflowSpec,
)
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


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
            # Use default (empty) for event_schema
        )
    ]

    _register_triggers(triggers, env)

    # Should use default schema with additionalProperties: True
    # This allows property access on the trigger in expressions like ${t1.anyProperty}
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
    # 'vars' is always registered for global variable support
    assert len(env.as_dict()) == 1
    assert "vars" in env.as_dict()


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


def test_build_type_environment_with_agent_node():
    """Test building type environment with agent node."""
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            AgentNode(
                id="llm1",
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
            AgentNode(
                id="llm1",
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


# =============================================================================
# Loop Variable Type Inference Tests
# =============================================================================


def test_for_each_loop_variable_infers_type_from_typed_array():
    """Test that for_each loop variables inherit type from source array's items schema.

    When the items expression references an array with a defined items schema,
    the loop variable should inherit that schema for proper type-safe property access.
    """
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="trigger1",
                key="test.trigger",
                mode="polling",
                event_schema={
                    "type": "object",
                    "properties": {
                        "organizations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "contacts": {"type": "array"},
                                    "id": {"type": "integer"}
                                }
                            }
                        }
                    }
                },
            )
        ],
        nodes=[
            ForEachNode(
                id="loop_orgs",
                type="for_each",
                items="${trigger1.organizations}",
                item_var="org",
                index_var="idx"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Loop variable should have inferred schema from array items
    assert "org" in symbols
    org_schema = symbols["org"]
    assert org_schema["type"] == "object"
    assert "properties" in org_schema
    assert "name" in org_schema["properties"]
    assert "contacts" in org_schema["properties"]
    assert "id" in org_schema["properties"]

    # Property types should be preserved
    assert org_schema["properties"]["name"]["type"] == "string"
    assert org_schema["properties"]["contacts"]["type"] == "array"
    assert org_schema["properties"]["id"]["type"] == "integer"

    # Index variable should be integer
    assert "idx" in symbols
    assert symbols["idx"]["type"] == "integer"


def test_for_each_loop_variable_uses_permissive_fallback_when_no_items_schema():
    """Test that loop variables use permissive schema when source has no items schema.

    When the source array doesn't have an items schema defined, the loop variable
    should use a permissive schema that allows any property access.
    """
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="trigger1",
                key="test.trigger",
                mode="polling",
                event_schema={
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"}  # Array without items schema
                    }
                },
            )
        ],
        nodes=[
            ForEachNode(
                id="loop1",
                type="for_each",
                items="${trigger1.items}",
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Loop variable should have permissive schema (additionalProperties: True)
    assert "item" in symbols
    item_schema = symbols["item"]
    assert item_schema["type"] == "object"
    # additionalProperties should be True to allow any property access
    assert item_schema.get("additionalProperties") is True


def test_for_each_loop_variable_uses_permissive_fallback_for_unknown_source():
    """Test that loop variables use permissive schema when source is unknown.

    When the items expression references an unknown symbol, the loop variable
    should still use a permissive schema.
    """
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            ForEachNode(
                id="loop1",
                type="for_each",
                items="${unknown_items}",  # References unknown symbol
                item_var="item",
                index_var="index"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Loop variable should have permissive schema as fallback
    assert "item" in symbols
    item_schema = symbols["item"]
    assert item_schema["type"] == "object"
    assert item_schema.get("additionalProperties") is True


def test_for_each_loop_variable_infers_type_from_agent_node_output():
    """Test that for_each loop variables inherit type from agent node's JSON output schema.

    This is the key use case from the developer review - when an agent node outputs
    a structured JSON array, loop variables should be able to access its properties.
    """
    spec = WorkflowSpec(
        version="2",
        triggers=[],
        nodes=[
            AgentNode(
                id="prepare_data",
                inputs={"model": "gpt-4", "prompt": "Generate organizations"},
                outputs=OutputContract(
                    mode=OutputMode.json,
                    schema={
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "organizations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                            "active": {"type": "boolean"}
                                        },
                                        "required": ["name", "email", "active"],
                                    }
                                }
                            },
                            "required": ["organizations"],
                        }
                    }
                )
            ),
            ForEachNode(
                id="loop_orgs",
                type="for_each",
                items="${prepare_data.organizations}",
                item_var="org",
                index_var="i"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Loop variable should have inferred schema from LLM output
    assert "org" in symbols
    org_schema = symbols["org"]
    assert org_schema["type"] == "object"
    assert "properties" in org_schema
    assert "name" in org_schema["properties"]
    assert "email" in org_schema["properties"]
    assert "active" in org_schema["properties"]

    # Property types should be preserved
    assert org_schema["properties"]["name"]["type"] == "string"
    assert org_schema["properties"]["email"]["type"] == "string"
    assert org_schema["properties"]["active"]["type"] == "boolean"


def test_for_each_loop_variable_infers_type_from_nested_property():
    """Test type inference works for nested property paths like ${node.data.items}."""
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="webhook",
                key="test.trigger",
                mode="webhook",
                event_schema={
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "object",
                            "properties": {
                                "data": {
                                    "type": "object",
                                    "properties": {
                                        "users": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "username": {"type": "string"},
                                                    "age": {"type": "number"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
            )
        ],
        nodes=[
            ForEachNode(
                id="loop_users",
                type="for_each",
                items="${webhook.response.data.users}",
                item_var="user",
                index_var="idx"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Loop variable should have inferred schema from deeply nested path
    assert "user" in symbols
    user_schema = symbols["user"]
    assert user_schema["type"] == "object"
    assert "properties" in user_schema
    assert "username" in user_schema["properties"]
    assert "age" in user_schema["properties"]
    assert user_schema["properties"]["username"]["type"] == "string"
    assert user_schema["properties"]["age"]["type"] == "number"


def test_for_each_multiple_loops_with_different_item_types():
    """Test that multiple for_each loops can have different inferred types."""
    spec = WorkflowSpec(
        version="2",
        triggers=[
            TriggerSpec(
                id="trigger1",
                key="test.trigger",
                mode="polling",
                event_schema={
                    "type": "object",
                    "properties": {
                        "users": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}}
                            }
                        },
                        "products": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"price": {"type": "number"}}
                            }
                        }
                    }
                },
            )
        ],
        nodes=[
            ForEachNode(
                id="loop_users",
                type="for_each",
                items="${trigger1.users}",
                item_var="user",
                index_var="user_idx"
            ),
            ForEachNode(
                id="loop_products",
                type="for_each",
                items="${trigger1.products}",
                item_var="product",
                index_var="product_idx"
            )
        ],
        edges=[]
    )
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    env = build_type_environment(spec, schema_registry=schema_registry, tool_registry=tool_registry)

    symbols = env.as_dict()

    # Each loop variable should have its own inferred schema
    assert "user" in symbols
    assert symbols["user"]["properties"]["name"]["type"] == "string"
    assert "price" not in symbols["user"].get("properties", {})

    assert "product" in symbols
    assert symbols["product"]["properties"]["price"]["type"] == "number"
    assert "name" not in symbols["product"].get("properties", {})
