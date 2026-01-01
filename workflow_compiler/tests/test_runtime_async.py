from __future__ import annotations

import pytest

from workflow_compiler.compiler.emit_langgraph import emit_langgraph
from workflow_compiler.compiler.lower_control_flow import build_execution_plan
from workflow_compiler.compiler.parse import parse_workflow_spec
from workflow_compiler.compiler.type_env import build_type_environment
from workflow_compiler.compiler.validate_refs import validate_references
from workflow_compiler.registry.model_registry import ModelRegistry
from workflow_compiler.registry.tool_registry import ToolDefinition, ToolRegistry
from workflow_compiler.runtime.execution import CompiledWorkflow
from workflow_compiler.runtime.nodes import NodeRuntime, RuntimeServices
from workflow_compiler.schema.schema_registry import SchemaRegistry


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

    spec = {
        "version": "1",
        "inputs": {
            "flag": {"type": "boolean", "required": True},
            "payload": {"type": "string", "required": True},
        },
        "nodes": [
            {
                "id": "conditional",
                "type": "if",
                "condition": "${inputs.flag}",
                "then": [
                    {
                        "id": "call_tool",
                        "type": "tool",
                        "tool": "test.echo",
                        "in": {"message": "${inputs.payload}"},
                        "out": "tool_result",
                    }
                ],
                "else": [],
            }
        ],
        "output": "${tool_result}",
    }

    compiled = await _compile_workflow(spec, [tool_def])
    result = await compiled.ainvoke(
        inputs={"flag": True, "payload": "hello"},
        config=None,
        context=None,
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

    spec = {
        "version": "1",
        "inputs": {},
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
                "body": [
                    {
                        "id": "loop_tool",
                        "type": "tool",
                        "tool": "test.loop_echo",
                        "in": {"message": "${item}"},
                        "out": "loop_result",
                    }
                ],
                "out": "aggregated",
            },
        ],
        "output": "${aggregated}",
    }

    compiled = await _compile_workflow(spec, [tool_def])
    result = await compiled.ainvoke(inputs={}, config=None, context=None)

    assert result["aggregated"] == [
        {"echo": "alpha-processed"},
        {"echo": "beta-processed"},
    ]
    assert async_calls == ["alpha", "beta"]

