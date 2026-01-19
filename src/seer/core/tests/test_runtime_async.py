from __future__ import annotations

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.schema_registry import SchemaRegistry


async def _compile_workflow(spec_payload: dict, tool_defs: list[ToolDefinition]) -> CompiledWorkflow:
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    for tool in tool_defs:
        tool_registry.register(tool)
    model_registry = ModelRegistry()

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
async def test_if_branch_tool_runs_async_handler() -> None:
    async_calls: list[str] = []

    def sync_handler(inputs, config, context):
        raise AssertionError("synchronous handler should never run in async execution")

    async def async_handler(inputs, config, context):
        async_calls.append(inputs["message"])
        return {"echo": inputs["message"]}

    tool_def = ToolDefinition(
        name="test.echo",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
            "additionalProperties": False,
        },
        handler=sync_handler,
        async_handler=async_handler,
    )

    # V2 workflow with explicit edges for if/else branching
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.trigger",
                "title": "TestTrigger",
                "provider": "test",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "flag": {"type": "boolean"},
                                    "payload": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "conditional",
                "type": "if",
                "condition": "${TestTrigger.data.flag}",
            },
            {
                "id": "call_tool",
                "type": "tool",
                "tool": "test.echo",
                "in": {"message": "${TestTrigger.data.payload}"},
                "out": "tool_result",
            },
        ],
        "edges": [
            {"id": "e0", "source": "test_trigger", "target": "conditional", "type": "trigger"},
            {"id": "e1", "source": "conditional", "target": "call_tool", "type": "conditional_true"},
        ],
    }

    compiled = await _compile_workflow(spec, [tool_def])
    # Pass trigger envelope with event data and trigger_key for routing
    trigger_envelope = {"trigger_key": "test.trigger", "title": "TestTrigger", "data": {"flag": True, "payload": "hello"}}
    result = await compiled.ainvoke(
        config=None,
        context=None,
        trigger=trigger_envelope,
    )

    assert result["tool_result"]["echo"] == "hello"
    assert async_calls == ["hello"]


@pytest.mark.asyncio
async def test_for_each_body_tools_use_async_handler() -> None:
    async_calls: list[str] = []

    def sync_handler(inputs, config, context):
        raise AssertionError("synchronous handler should never run in async execution")

    async def async_handler(inputs, config, context):
        async_calls.append(inputs["message"])
        return {"echo": f"{inputs['message']}-processed"}

    tool_def = ToolDefinition(
        name="test.loop_echo",
        version="v1",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
            "additionalProperties": False,
        },
        handler=sync_handler,
        async_handler=async_handler,
    )

    # V2 workflow with explicit edges for loop control flow
    # Note: With edge-based loops, aggregation works differently - we verify each
    # iteration runs and the final result is available
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "loop_trigger",
                "key": "loop.trigger",
                "title": "LoopTrigger",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "items_seed",
                "type": "task",
                "kind": "set",
                "value": ["alpha", "beta"],
                "out": "items_state",
            },
            {
                "id": "loop",
                "type": "for_each",
                "items": "${items_state}",
            },
            {
                "id": "loop_tool",
                "type": "tool",
                "tool": "test.loop_echo",
                "in": {"message": "${item}"},
                "out": "loop_result",
            },
            {
                "id": "done",
                "type": "task",
                "kind": "set",
                "value": "finished",
                "out": "status",
            },
        ],
        "edges": [
            {"id": "e0", "source": "loop_trigger", "target": "items_seed", "type": "trigger"},
            {"id": "e1", "source": "items_seed", "target": "loop", "type": "default"},
            {"id": "e2", "source": "loop", "target": "loop_tool", "type": "loop_body"},
            {"id": "e3", "source": "loop_tool", "target": "loop", "type": "default"},
            {"id": "e4", "source": "loop", "target": "done", "type": "loop_exit"},
        ],
    }

    compiled = await _compile_workflow(spec, [tool_def])
    trigger_envelope = {"trigger_key": "loop.trigger", "title": "LoopTrigger"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Verify both loop iterations ran with async handler
    assert async_calls == ["alpha", "beta"]
    # Verify the loop completed and reached the done node
    assert result["status"] == "finished"
    # Verify the last iteration's result is in state
    assert result["loop_result"]["echo"] == "beta-processed"


@pytest.mark.asyncio
async def test_trigger_routes_to_correct_node() -> None:
    """Single trigger edge routes to the correct target node."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "webhook_received",
                "key": "webhook.received",
                "title": "Webhook",
                "provider": "webhook",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            }
        ],
        "nodes": [
            {
                "id": "handler",
                "type": "task",
                "kind": "set",
                "value": "webhook_handled",
                "out": "result",
            },
        ],
        "edges": [
            {"id": "e1", "source": "webhook_received", "target": "handler", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    trigger_envelope = {"trigger_key": "webhook.received", "title": "Webhook"}
    result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["result"] == "webhook_handled"


@pytest.mark.asyncio
async def test_multiple_triggers_route_to_different_nodes() -> None:
    """Multiple triggers route to their respective target nodes based on trigger_key."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "github_push",
                "key": "github.push",
                "title": "Push",
                "provider": "github",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            },
            {
                "id": "github_pr",
                "key": "github.pr",
                "title": "PR",
                "provider": "github",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            },
        ],
        "nodes": [
            {
                "id": "handle_push",
                "type": "task",
                "kind": "set",
                "value": "push_handled",
                "out": "result",
            },
            {
                "id": "handle_pr",
                "type": "task",
                "kind": "set",
                "value": "pr_handled",
                "out": "result",
            },
        ],
        "edges": [
            {"id": "e1", "source": "github_push", "target": "handle_push", "type": "trigger"},
            {"id": "e2", "source": "github_pr", "target": "handle_pr", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow(spec, [])

    # Test push trigger routes to handle_push
    result = await compiled.ainvoke(
        config=None, context=None,
        trigger={"trigger_id": "github_push", "trigger_key": "github.push", "title": "Push"}
    )
    assert result["result"] == "push_handled"

    # Test PR trigger routes to handle_pr
    result = await compiled.ainvoke(
        config=None, context=None,
        trigger={"trigger_id": "github_pr", "trigger_key": "github.pr", "title": "PR"}
    )
    assert result["result"] == "pr_handled"


@pytest.mark.asyncio
async def test_multiple_triggers_route_to_same_node() -> None:
    """Multiple triggers can route to the same target node."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_a",
                "key": "trigger_a",
                "title": "A",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            },
            {
                "id": "trigger_b",
                "key": "trigger_b",
                "title": "B",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            },
        ],
        "nodes": [
            {
                "id": "shared_handler",
                "type": "task",
                "kind": "set",
                "value": "shared_executed",
                "out": "result",
            },
        ],
        "edges": [
            {"id": "e1", "source": "trigger_a", "target": "shared_handler", "type": "trigger"},
            {"id": "e2", "source": "trigger_b", "target": "shared_handler", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow(spec, [])

    # Both triggers should route to the same node
    for trigger_key, title in [("trigger_a", "A"), ("trigger_b", "B")]:
        result = await compiled.ainvoke(
            config=None, context=None,
            trigger={"trigger_key": trigger_key, "title": title}
        )
        assert result["result"] == "shared_executed"


@pytest.mark.asyncio
async def test_unknown_trigger_key_fallback() -> None:
    """Unknown trigger key falls back to first trigger's target."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "known_trigger",
                "key": "known_trigger",
                "title": "Known",
                "provider": "test",
                "mode": "webhook",
                "schemas": {"event": {"type": "object"}},
            },
        ],
        "nodes": [
            {
                "id": "handler",
                "type": "task",
                "kind": "set",
                "value": "handled",
                "out": "result",
            },
        ],
        "edges": [
            {"id": "e1", "source": "known_trigger", "target": "handler", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow(spec, [])

    # Unknown trigger key should fall back to the first target
    result = await compiled.ainvoke(
        config=None, context=None,
        trigger={"trigger_key": "unknown_trigger", "title": "Known"}
    )
    assert result["result"] == "handled"


@pytest.mark.asyncio
async def test_trigger_data_accessible_after_routing() -> None:
    """Trigger data is accessible via ${trigger.*} expressions after routing."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "data_trigger",
                "key": "data.trigger",
                "title": "Data",
                "provider": "test",
                "mode": "webhook",
                "schemas": {
                    "event": {
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        ],
        "nodes": [
            {
                "id": "echo",
                "type": "task",
                "kind": "set",
                "value": "${Data.data.message}",
                "out": "result",
            },
        ],
        "edges": [
            {"id": "e1", "source": "data_trigger", "target": "echo", "type": "trigger"},
        ],
    }

    compiled = await _compile_workflow(spec, [])
    result = await compiled.ainvoke(
        config=None, context=None,
        trigger={"trigger_key": "data.trigger", "title": "Data", "data": {"message": "hello world"}}
    )

    assert result["result"] == "hello world"
