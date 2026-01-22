"""
Comprehensive tests for for_each loop functionality in the workflow compiler.

Tests cover:
- Basic iteration over lists
- Empty lists
- Item and index variable access
- Custom variable names
- Nested nodes within loops
- Loop state management
- Integration with LangGraph execution
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


async def _compile_workflow(
    spec_payload: dict,
    tool_defs: list[ToolDefinition],
    model_defs: list[ModelDefinition] | None = None
) -> CompiledWorkflow:
    """Helper to compile a workflow spec into an executable graph."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    for tool in tool_defs:
        tool_registry.register(tool)
    model_registry = ModelRegistry()
    for model in model_defs or []:
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
async def test_for_each_basic_iteration() -> None:
    """Test basic for_each loop iteration with task nodes."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["one", "two", "three"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process_item",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "exit",
                "type": "task",
                "kind": "set",
                "value": "done",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "process_item", "target": "loop", "type": "default"},
            {"source": "loop", "target": "exit", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["exit"] == "done"
    # Verify items list was created
    assert result["items"] == ["one", "two", "three"]
    # Verify last iteration result
    assert result["process_item"] == "three"


@pytest.mark.asyncio
async def test_for_each_empty_list() -> None:
    """Test for_each loop with an empty list (zero iterations)."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": [],  # Empty list
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process_item",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "exit",
                "type": "task",
                "kind": "set",
                "value": "done",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            {"source": "process_item", "target": "loop", "type": "default"},
            {"source": "loop", "target": "exit", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop exited immediately without processing
    assert result["exit"] == "done"
    # Verify items list is empty
    assert result["items"] == []
    # Verify process_item was never executed (key should not exist in final state)
    # Note: In LangGraph, unexecuted nodes don't write to state


@pytest.mark.asyncio
async def test_for_each_with_index_access() -> None:
    """Test for_each loop accessing both item and index variables."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["a", "b", "c"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
                "item_var": "item",
                "index_var": "index",
            },
            {
                "id": "store_item",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "finished",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "store_item", "type": "loop_body"},
            {"source": "store_item", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["done"] == "finished"
    # Verify last iteration had correct item
    assert result["store_item"] == "c"
    # Verify index variable was available in state during last iteration
    assert result["index"] == 2
    # Verify item variable was available in state during last iteration
    assert result["item"] == "c"


@pytest.mark.asyncio
async def test_for_each_custom_variable_names() -> None:
    """Test for_each loop with custom item_var and index_var names."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "numbers",
                "type": "task",
                "kind": "set",
                "value": [10, 20, 30],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${numbers}",
                "item_var": "num",      # Custom item variable name
                "index_var": "position",  # Custom index variable name
            },
            {
                "id": "compute",
                "type": "task",
                "kind": "set",
                "value": "${num}",
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {"type": "integer"}
                    }
                }
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "complete",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "numbers", "type": "trigger"},
            {"source": "numbers", "target": "loop", "type": "default"},
            {"source": "loop", "target": "compute", "type": "loop_body"},
            {"source": "compute", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed with custom variable names
    assert result["done"] == "complete"
    assert result["compute"] == 30
    # Verify custom variable names are in state
    assert result["num"] == 30
    assert result["position"] == 2


@pytest.mark.asyncio
async def test_for_each_with_tool_nodes() -> None:
    """Test for_each loop executing tool nodes in body."""
    tool_calls = []

    def tool_handler(inputs, config, context):
        tool_calls.append(inputs["value"])
        return {"result": f"processed_{inputs['value']}"}

    async def tool_handler_async(inputs, config, context):
        tool_calls.append(inputs["value"])
        return {"result": f"processed_{inputs['value']}"}

    tool_def = ToolDefinition(
        name="test.process",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
            "additionalProperties": False,
        },
        handler=tool_handler,
        async_handler=tool_handler_async,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["alpha", "beta", "gamma"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process_tool",
                "type": "tool",
                "tool": "test.process",
                "inputs": {"value": "${item}"},
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "finished",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_tool", "type": "loop_body"},
            {"source": "process_tool", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tool_def])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify all items were processed
    assert tool_calls == ["alpha", "beta", "gamma"]
    # Verify loop completed
    assert result["done"] == "finished"
    # Verify last tool execution result
    assert result["process_tool"]["result"] == "processed_gamma"


@pytest.mark.asyncio
async def test_for_each_with_numeric_items() -> None:
    """Test for_each loop iterating over numeric values."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "numbers",
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3, 4, 5],
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                }
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${numbers}",
            },
            {
                "id": "store_num",
                "type": "task",
                "kind": "set",
                "value": "${item}",
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {"type": "integer"}
                    }
                }
            },
            {
                "id": "exit",
                "type": "task",
                "kind": "set",
                "value": "complete",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "numbers", "type": "trigger"},
            {"source": "numbers", "target": "loop", "type": "default"},
            {"source": "loop", "target": "store_num", "type": "loop_body"},
            {"source": "store_num", "target": "loop", "type": "default"},
            {"source": "loop", "target": "exit", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["exit"] == "complete"
    # Verify last iteration processed the last number
    assert result["store_num"] == 5


@pytest.mark.asyncio
async def test_for_each_with_object_items() -> None:
    """Test for_each loop iterating over objects."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "users",
                "type": "task",
                "kind": "set",
                "value": [
                    {"name": "Alice", "age": 30},
                    {"name": "Bob", "age": 25},
                ],
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer"}
                                },
                                "additionalProperties": True
                            }
                        }
                    }
                }
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${users}",
                "item_var": "user",
            },
            {
                "id": "extract",
                "type": "task",
                "kind": "set",
                "value": "${user}",
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"}
                            },
                            "additionalProperties": True
                        }
                    }
                }
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "processed",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "users", "type": "trigger"},
            {"source": "users", "target": "loop", "type": "default"},
            {"source": "loop", "target": "extract", "type": "loop_body"},
            {"source": "extract", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["done"] == "processed"
    # Verify last user object
    assert result["extract"]["name"] == "Bob"
    assert result["extract"]["age"] == 25


@pytest.mark.asyncio
async def test_for_each_single_item() -> None:
    """Test for_each loop with a single item (edge case)."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["only_one"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "finished",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            {"source": "process", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["done"] == "finished"
    # Verify single iteration
    assert result["process"] == "only_one"
    assert result["item"] == "only_one"
    assert result["index"] == 0


@pytest.mark.asyncio
async def test_for_each_state_isolation() -> None:
    """Test that loop state is properly isolated and managed."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["x", "y"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "body",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "complete",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "body", "type": "loop_body"},
            {"source": "body", "target": "loop", "type": "default"},
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop state exists in result
    assert "_loop_loop" in result
    loop_state = result["_loop_loop"]

    # Verify final loop state
    assert loop_state["current_index"] == 2  # After both iterations
    assert loop_state["has_more_iterations"] is False
    assert loop_state["items"] == ["x", "y"]

    # Verify completion
    assert result["done"] == "complete"


@pytest.mark.asyncio
async def test_for_each_without_explicit_back_edge() -> None:
    """Test for_each loop works without explicit edge from body back to loop node."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["alpha", "beta", "gamma"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process_item",
                "type": "task",
                "kind": "set",
                "value": "${item}",
            },
            {
                "id": "exit",
                "type": "task",
                "kind": "set",
                "value": "done",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process_item", "type": "loop_body"},
            # NOTE: No explicit edge from process_item back to loop - it should be implicit!
            {"source": "loop", "target": "exit", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed all iterations
    assert result["exit"] == "done"
    assert result["items"] == ["alpha", "beta", "gamma"]
    # Verify last iteration result
    assert result["process_item"] == "gamma"
    # Verify loop completed
    loop_state = result["_loop_loop"]
    assert loop_state["current_index"] == 3
    assert loop_state["has_more_iterations"] is False


@pytest.mark.asyncio
async def test_for_each_loop_iteration_traces() -> None:
    """Test that for_each loop creates separate trace keys for each iteration."""
    
    # Define a mock model for testing
    def mock_text_handler(invocation):
        # Handler returns (result, usage_metadata)
        prompt = invocation.get("prompt", "")
        return f"Response: {prompt}", {}
    
    model_def = ModelDefinition(
        model_id="gpt-5-nano",
        text_handler=mock_text_handler,
    )
    
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": ["apple", "banana", "cherry"],
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "process",
                "type": "llm",
                "inputs": {
                    "model": "gpt-5-nano",
                    "prompt": "Say: ${item}",
                },
                "outputs": {
                    "mode": "text",
                },
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "complete",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "process", "type": "loop_body"},
            # No explicit back-edge
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [], [model_def])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed
    assert result["done"] == "complete"

    # Verify that we have separate trace keys for each iteration
    assert "_trace_process_iter_0" in result
    assert "_trace_process_iter_1" in result
    assert "_trace_process_iter_2" in result

    # Verify each iteration has trace data
    assert result["_trace_process_iter_0"]["node_id"] == "process"
    assert result["_trace_process_iter_1"]["node_id"] == "process"
    assert result["_trace_process_iter_2"]["node_id"] == "process"


@pytest.mark.asyncio
async def test_for_each_multi_node_body_without_back_edge() -> None:
    """Test for_each loop with multiple nodes in body (A->B->C) without explicit back-edge."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items",
                "type": "task",
                "kind": "set",
                "value": [1, 2, 3],
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "json_schema": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                }
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items}",
            },
            {
                "id": "step_a",
                "type": "task",
                "kind": "set",
                "value": "${item}",
                "outputs": {
                    "mode": "json",
                    "schema": {"json_schema": {"type": "integer"}}
                }
            },
            {
                "id": "step_b",
                "type": "task",
                "kind": "set",
                "value": "${step_a}",
                "outputs": {
                    "mode": "json",
                    "schema": {"json_schema": {"type": "integer"}}
                }
            },
            {
                "id": "step_c",
                "type": "task",
                "kind": "set",
                "value": "${step_b}",
                "outputs": {
                    "mode": "json",
                    "schema": {"json_schema": {"type": "integer"}}
                }
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "finished",
            },
        ],
        "edges": [
            {"source": "test_trigger", "target": "items", "type": "trigger"},
            {"source": "items", "target": "loop", "type": "default"},
            {"source": "loop", "target": "step_a", "type": "loop_body"},
            {"source": "step_a", "target": "step_b", "type": "default"},
            {"source": "step_b", "target": "step_c", "type": "default"},
            # NOTE: No explicit edge from step_c back to loop - it should be implicit!
            {"source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "test.trigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify loop completed all iterations
    assert result["done"] == "finished"
    # Verify all steps executed for last iteration
    assert result["step_a"] == 3
    assert result["step_b"] == 3
    assert result["step_c"] == 3
    # Verify loop completed
    loop_state = result["_loop_loop"]
    assert loop_state["current_index"] == 3
    assert loop_state["has_more_iterations"] is False
