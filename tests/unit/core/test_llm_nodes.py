# pylint: disable=unused-argument
# Reason: Mock handlers require specific function signatures even if not all params are used
"""
Tests for LLM node execution in workflows.

These tests verify that LLM nodes execute correctly with both text and JSON output modes.
This test file was created after discovering a bug where node.out was referenced instead
of node.id in the runtime execution code.
"""
from __future__ import annotations

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


def _create_mock_tool() -> ToolDefinition:
    """Create a mock test.tool that simply returns its input value."""
    def handler(inputs, config, context):
        return inputs.get("value", "")

    async def async_handler(inputs, config, context):
        return inputs.get("value", "")

    return ToolDefinition(
        name="test.tool",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": ["string", "array", "object", "number", "boolean", "null"]}},
            "additionalProperties": False,
        },
        output_schema={"type": ["string", "array", "object", "number", "boolean", "null"]},
        handler=handler,
        async_handler=async_handler,
    )


async def _compile_workflow_with_models(
    spec_payload: dict, model_defs: list[ModelDefinition], tool_defs: list[ToolDefinition] | None = None
) -> CompiledWorkflow:
    """Helper to compile a workflow with model registry."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    # Register tools
    for tool in tool_defs or []:
        tool_registry.register(tool)
    model_registry = ModelRegistry()

    # Register models
    for model in model_defs:
        model_registry.register(model)

    spec = parse_workflow_spec(spec_payload)
    type_env = build_type_environment(
        spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )
    validate_references(spec, type_env)
    plan = build_execution_plan(spec)

    runtime = NodeRuntime(
        RuntimeServices(
            schema_registry=schema_registry,
            tool_registry=tool_registry,
            model_registry=model_registry,
            type_env=type_env,
        )
    )
    graph = await emit_langgraph(plan, runtime)
    return CompiledWorkflow(
        spec=spec,
        type_env=type_env.as_dict(),
        graph=graph,
        runtime=runtime,
    )


@pytest.mark.asyncio
async def test_llm_node_with_text_output() -> None:
    """Test LLM node execution with text output mode."""

    def mock_text_handler(invocation):
        # Handler returns (result, usage_metadata)
        prompt = invocation.get("prompt", "")
        return f"Mock response to: {prompt}", {}

    model_def = ModelDefinition(
        model_id="test-text-model",
        text_handler=mock_text_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "text_trigger",
                "key": "test.text",
                "title": "TextTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm_text",
                "type": "llm",
                "inputs": {
                    "model": "test-text-model",
                    "prompt": "Generate a greeting",
                },
                "outputs": {
                    "mode": "text",
                }
            }
        ],
        "edges": [
            {"source": "text_trigger", "target": "llm_text", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow_with_models(spec, [model_def])
    trigger_envelope = {"trigger_key": "test.text", "title": "TextTest"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Result should be stored under node.id
    assert "llm_text" in result
    assert result["llm_text"] == "Mock response to: Generate a greeting"


@pytest.mark.asyncio
async def test_llm_node_with_json_output() -> None:
    """
    Test LLM node execution with JSON output mode.

    This test would have caught the bug where node.out was referenced
    instead of node.id for schema lookup.
    """

    def mock_json_handler(invocation, schema):
        # Handler returns (result, usage_metadata)
        # Return structured data matching the schema
        return {
            "pet1": "Fluffy",
            "pet2": "Spot",
        }, {}

    model_def = ModelDefinition(
        model_id="gpt-5-mini",
        json_handler=mock_json_handler,
    )

    # This is the exact spec that was failing in production
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "json_trigger",
                "key": "test.json",
                "title": "JsonTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm-1",
                "type": "llm",
                "inputs": {
                    "model": "gpt-5-mini",
                    "prompt": "generate random pet names"
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "pet1": {
                                    "type": "string",
                                    "description": "name of pet1"
                                },
                                "pet2": {
                                    "type": "string",
                                    "description": "name of pet 2"
                                }
                            }
                        }
                    }
                }
            }
        ],
        "edges": [
            {"source": "json_trigger", "target": "llm-1", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow_with_models(spec, [model_def])
    trigger_envelope = {"trigger_key": "test.json", "title": "JsonTest"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Before the fix, this would fail with:
    # AttributeError: 'LLMNode' object has no attribute 'out'
    assert "llm-1" in result
    assert result["llm-1"]["pet1"] == "Fluffy"
    assert result["llm-1"]["pet2"] == "Spot"


@pytest.mark.asyncio
async def test_llm_node_output_used_in_next_node() -> None:
    """Test that LLM node output can be referenced by subsequent nodes."""

    def mock_text_handler(invocation):
        return "test-value", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "chained_trigger",
                "key": "test.chain",
                "title": "ChainTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm1",
                "type": "llm",
                "inputs": {
                    "model": "test-model",
                    "prompt": "First prompt",
                },
                "outputs": {
                    "mode": "text",
                }
            },
            {
                "id": "task1",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "${llm1}"},  # Reference LLM output by node.id
            }
        ],
        "edges": [
            {"source": "chained_trigger", "target": "llm1", "type": "trigger"},
            {"source": "llm1", "target": "task1", "type": "default"},
        ],
    }

    mock_tool = _create_mock_tool()
    compiled = await _compile_workflow_with_models(spec, [model_def], [mock_tool])
    trigger_envelope = {"trigger_key": "test.chain", "title": "ChainTest"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["llm1"] == "test-value"
    assert result["task1"] == "test-value"


@pytest.mark.asyncio
async def test_llm_node_in_conditional_branch() -> None:
    """Test LLM node execution within conditional branches."""

    def mock_json_handler(invocation, schema):
        return {"success": True, "message": "OK"}, {}

    model_def = ModelDefinition(
        model_id="test-model",
        json_handler=mock_json_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "cond_trigger",
                "key": "test.condition",
                "title": "CondTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "should_run": {"type": "boolean"},
                                },
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "condition",
                "type": "if",
                "condition": "${cond_trigger.data.should_run}",
            },
            {
                "id": "llm_on_true",
                "type": "llm",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Process when true",
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "message": {"type": "string"},
                            }
                        }
                    }
                }
            },
        ],
        "edges": [
            {"source": "cond_trigger", "target": "condition", "type": "trigger"},
            {"source": "condition", "target": "llm_on_true", "type": "conditional_true"},
        ],
    }

    compiled = await _compile_workflow_with_models(spec, [model_def])
    trigger_envelope = {
        "id": "cond_trigger",
        "trigger_id": "cond_trigger",
        "trigger_key": "test.condition",
        "title": "CondTest",
        "data": {"should_run": True}
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "llm_on_true" in result
    assert result["llm_on_true"]["success"] is True
    assert result["llm_on_true"]["message"] == "OK"


@pytest.mark.asyncio
async def test_llm_node_with_dynamic_prompt_from_trigger() -> None:
    """Test LLM node with prompt that references trigger data."""

    received_prompts = []

    def mock_text_handler(invocation):
        prompt = invocation.get("prompt", "")
        received_prompts.append(prompt)
        return f"Response to: {prompt}", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "dynamic_trigger",
                "key": "test.dynamic",
                "title": "DynamicTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "user_input": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "llm_dynamic",
                "type": "llm",
                "inputs": {
                    "model": "test-model",
                    "prompt": "User asked: ${dynamic_trigger.data.user_input}",
                },
                "outputs": {
                    "mode": "text",
                }
            }
        ],
        "edges": [
            {"source": "dynamic_trigger", "target": "llm_dynamic", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow_with_models(spec, [model_def])
    trigger_envelope = {
        "id": "dynamic_trigger",
        "trigger_id": "dynamic_trigger",
        "trigger_key": "test.dynamic",
        "title": "DynamicTest",
        "data": {"user_input": "What is the weather?"}
    }
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["llm_dynamic"] == "Response to: User asked: What is the weather?"
    assert received_prompts == ["User asked: What is the weather?"]


@pytest.mark.asyncio
async def test_multiple_llm_nodes_in_sequence() -> None:
    """Test multiple LLM nodes using outputs from previous ones."""

    call_order = []

    def mock_handler_1(invocation):
        call_order.append("llm1")
        return "first result", {}

    def mock_handler_2(invocation):
        call_order.append("llm2")
        prompt = invocation.get("prompt", "")
        return f"second result based on: {prompt}", {}

    model_def_1 = ModelDefinition(
        model_id="model-1",
        text_handler=mock_handler_1,
    )

    model_def_2 = ModelDefinition(
        model_id="model-2",
        text_handler=mock_handler_2,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "seq_trigger",
                "key": "test.sequence",
                "title": "SeqTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm_first",
                "type": "llm",
                "inputs": {
                    "model": "model-1",
                    "prompt": "Generate initial data",
                },
                "outputs": {
                    "mode": "text",
                }
            },
            {
                "id": "llm_second",
                "type": "llm",
                "inputs": {
                    "model": "model-2",
                    "prompt": "Process: ${llm_first}",
                },
                "outputs": {
                    "mode": "text",
                }
            }
        ],
        "edges": [
            {"source": "seq_trigger", "target": "llm_first", "type": "trigger"},
            {"source": "llm_first", "target": "llm_second", "type": "default"},
        ],
    }

    compiled = await _compile_workflow_with_models(spec, [model_def_1, model_def_2])
    trigger_envelope = {"trigger_key": "test.sequence", "title": "SeqTest"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert call_order == ["llm1", "llm2"]
    assert result["llm_first"] == "first result"
    assert result["llm_second"] == "second result based on: Process: first result"


@pytest.mark.asyncio
async def test_llm_node_array_schema_rejected() -> None:
    """Test that LLM nodes with array root type schemas are rejected at compile time.

    OpenAI structured outputs require root type to be 'object'. Array root types
    should fail with a clear error message during workflow compilation.
    """
    from seer.core.errors import TypeEnvironmentError

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "array_trigger",
                "key": "test.array",
                "title": "ArrayTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm-array",
                "type": "llm",
                "inputs": {
                    "model": "gpt-5-mini",
                    "prompt": "Generate a list of items"
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        ],
        "edges": [
            {"source": "array_trigger", "target": "llm-array", "type": "trigger"},
        ],
    }

    # Should fail at compile time with TypeEnvironmentError
    with pytest.raises(TypeEnvironmentError) as exc_info:
        schema_registry = SchemaRegistry()
        tool_registry = ToolRegistry()
        parsed_spec = parse_workflow_spec(spec)
        build_type_environment(
            parsed_spec,
            schema_registry=schema_registry,
            tool_registry=tool_registry,
        )

    # Verify error message is helpful
    error_message = str(exc_info.value)
    assert "llm-array" in error_message
    assert "array" in error_message.lower()
    assert "object" in error_message.lower()


@pytest.mark.asyncio
async def test_llm_node_object_with_array_property_allowed() -> None:
    """Test that LLM nodes with object root type containing array properties work fine.

    Arrays are only forbidden at the root level. Objects containing array properties
    should compile and execute successfully.
    """

    def mock_json_handler(invocation, schema):
        return {"items": [{"name": "Item1"}, {"name": "Item2"}]}, {}

    model_def = ModelDefinition(
        model_id="gpt-5-mini",
        json_handler=mock_json_handler,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "obj_array_trigger",
                "key": "test.obj_array",
                "title": "ObjArrayTest",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "llm-obj-array",
                "type": "llm",
                "inputs": {
                    "model": "gpt-5-mini",
                    "prompt": "Generate a list of items"
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        ],
        "edges": [
            {"source": "obj_array_trigger", "target": "llm-obj-array", "type": "trigger"},
        ],
    }

    # Should compile and execute successfully
    compiled = await _compile_workflow_with_models(spec, [model_def])
    trigger_envelope = {"trigger_key": "test.obj_array", "title": "ObjArrayTest"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "llm-obj-array" in result
    assert result["llm-obj-array"]["items"] == [{"name": "Item1"}, {"name": "Item2"}]
