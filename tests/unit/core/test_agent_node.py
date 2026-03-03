# pylint: disable=unused-argument
# Reason: Mock handlers require specific function signatures even if not all params are used
"""
Tests for Agent node execution in workflows.

These tests verify that Agent nodes execute correctly with state resolution,
tool binding, tracing, and output modes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.compiler.emit_langgraph import emit_langgraph
from seer.core.compiler.lower_control_flow import build_execution_plan
from seer.core.compiler.parse import parse_workflow_spec
from seer.core.compiler.type_env import build_type_environment
from seer.core.compiler.validate_refs import validate_references
from seer.core.errors import TypeEnvironmentError
from seer.core.registry.model_registry import ModelDefinition, ModelRegistry
from seer.core.registry.tool_registry import ToolDefinition, ToolRegistry
from seer.core.runtime.execution import CompiledWorkflow
from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
from seer.core.schema.models import AgentNode
from seer.core.schema.schema_registry import SchemaRegistry

pytestmark = pytest.mark.unit


# =============================================================================
# Schema Validation Tests
# =============================================================================


def test_agent_node_valid_basic():
    """Test basic valid AgentNode creation."""
    node = AgentNode(
        id="research_agent",
        inputs={
            "model": "gpt-4",
            "prompt": "Research the topic: ${topic}",
            "tools": ["web_search"],
            "max_iterations": 5,
        },
    )

    assert node.id == "research_agent"
    assert node.type == "agent"
    assert node.inputs["model"] == "gpt-4"
    assert node.inputs["tools"] == ["web_search"]


def test_agent_node_valid_with_tool_objects():
    """Test AgentNode with tool objects containing connection_id."""
    node = AgentNode(
        id="email_agent",
        inputs={
            "model": "gpt-4",
            "prompt": "Send an email to ${recipient}",
            "tools": [
                "extract_email",
                {"name": "gmail_send_email", "connection_id": 42},
            ],
        },
    )

    tools = node.inputs["tools"]
    assert isinstance(tools, list)
    assert len(tools) == 2
    assert tools[0] == "extract_email"
    assert isinstance(tools[1], dict)
    assert tools[1]["name"] == "gmail_send_email"


def test_agent_node_missing_model_raises():
    """Test that AgentNode without model raises validation error."""
    with pytest.raises(ValueError, match="requires.*model"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "prompt": "Do something",
            },
        )


def test_agent_node_missing_prompt_raises():
    """Test that AgentNode without prompt raises validation error."""
    with pytest.raises(ValueError, match="requires.*prompt"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "model": "gpt-4",
            },
        )


def test_agent_node_invalid_tool_format_raises():
    """Test that AgentNode with invalid tool format raises validation error."""
    with pytest.raises(ValueError, match="must be string or"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "model": "gpt-4",
                "prompt": "Do something",
                "tools": [123],  # Invalid: not string or dict
            },
        )


def test_agent_node_tool_object_missing_name_raises():
    """Test that AgentNode with tool object missing name raises validation error."""
    with pytest.raises(ValueError, match="must have 'name' field"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "model": "gpt-4",
                "prompt": "Do something",
                "tools": [{"connection_id": 42}],  # Missing 'name'
            },
        )


def test_agent_node_invalid_max_iterations_raises():
    """Test that AgentNode with invalid max_iterations raises validation error."""
    with pytest.raises(ValueError, match="max_iterations.*positive integer"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "model": "gpt-4",
                "prompt": "Do something",
                "max_iterations": -1,
            },
        )


def test_agent_node_tools_not_list_raises():
    """Test that AgentNode with non-list tools raises validation error."""
    with pytest.raises(ValueError, match="tools.*must be a list"):
        AgentNode(
            id="invalid_agent",
            inputs={
                "model": "gpt-4",
                "prompt": "Do something",
                "tools": "web_search",  # Should be a list
            },
        )


# =============================================================================
# Type Environment Registration Tests
# =============================================================================


def test_agent_node_type_environment_registration():
    """Test that AgentNode registers output schema in type environment."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.agent",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "agent1",
                "type": "agent",
                "inputs": {
                    "model": "gpt-4",
                    "prompt": "Research ${test_trigger}",
                    "tools": [],
                },
                "outputs": {"mode": "text"},
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "agent1", "type": "trigger"},
        ],
    }

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    parsed_spec = parse_workflow_spec(spec)

    type_env = build_type_environment(
        parsed_spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    # Agent with text output should register string schema
    assert type_env.get("agent1") == {"type": "string"}


def test_agent_node_json_output_registration():
    """Test that AgentNode with JSON output registers object schema."""
    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "test_trigger",
                "key": "test.agent",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "agent1",
                "type": "agent",
                "inputs": {
                    "model": "gpt-4",
                    "prompt": "Extract data",
                    "tools": [],
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "integer"},
                            },
                        }
                    },
                },
            }
        ],
        "edges": [
            {"source": "test_trigger", "target": "agent1", "type": "trigger"},
        ],
    }

    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()
    parsed_spec = parse_workflow_spec(spec)

    type_env = build_type_environment(
        parsed_spec,
        schema_registry=schema_registry,
        tool_registry=tool_registry,
    )

    schema = type_env.get("agent1")
    assert schema["type"] == "object"
    assert "name" in schema["properties"]


# =============================================================================
# Execution Tests
# =============================================================================


def _create_mock_tool_for_agent() -> ToolDefinition:
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


async def _compile_agent_workflow(
    spec_payload: dict,
    model_defs: list[ModelDefinition],
    tool_defs: list[ToolDefinition] | None = None,
) -> CompiledWorkflow:
    """Helper to compile a workflow with model registry."""
    schema_registry = SchemaRegistry()
    tool_registry = ToolRegistry()

    for tool in tool_defs or []:
        tool_registry.register(tool)

    model_registry = ModelRegistry()
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
async def test_agent_node_basic_execution():
    """Test basic AgentNode execution with mocked react_agent."""
    from langchain_core.messages import AIMessage, HumanMessage

    # Create mock chat model that returns a simple response
    mock_chat_model = MagicMock()

    def mock_text_handler(invocation):
        return "Mock response", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "agent_trigger",
                "key": "test.agent",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "agent1",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Research something",
                    "tools": [],
                    "max_iterations": 3,
                },
                "outputs": {"mode": "text"},
            }
        ],
        "edges": [
            {"source": "agent_trigger", "target": "agent1", "type": "trigger"},
        ],
    }

    # Mock create_agent to return a simple agent
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Research something"),
            AIMessage(content="I found the research results."),
        ]
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def])
        trigger_envelope = {"trigger_key": "test.agent"}
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Result should be stored under node.id
    assert "agent1" in result
    assert result["agent1"] == "I found the research results."

    # Trace should be present
    assert "_trace_agent1" in result
    trace = result["_trace_agent1"]
    assert trace["node_type"] == "agent"
    assert trace["status"] == "succeeded"
    assert "steps" in trace


@pytest.mark.asyncio
async def test_agent_node_with_tool_calls():
    """Test AgentNode execution that includes tool calls."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    mock_chat_model = MagicMock()

    def mock_text_handler(invocation):
        return "Mock response", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "tool_trigger",
                "key": "test.tool_agent",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "tool_agent",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Use tools to find data",
                    "tools": [],  # Empty for this test - we mock the agent directly
                    "max_iterations": 5,
                },
                "outputs": {"mode": "text"},
            }
        ],
        "edges": [
            {"source": "tool_trigger", "target": "tool_agent", "type": "trigger"},
        ],
    }

    # Create AI message with tool calls
    ai_msg_with_tools = AIMessage(
        content="I need to search for data.",
        tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "test"}}],
    )

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Use tools to find data"),
            ai_msg_with_tools,
            ToolMessage(content="Search results: test data", name="web_search", tool_call_id="call_1"),
            AIMessage(content="Based on the search, I found test data."),
        ]
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def])
        trigger_envelope = {"trigger_key": "test.tool_agent"}
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "tool_agent" in result
    assert result["tool_agent"] == "Based on the search, I found test data."

    # Check trace has tool call steps
    trace = result["_trace_tool_agent"]
    assert len(trace["steps"]) > 0

    # Find reasoning step with tool calls
    tool_call_step = next(
        (s for s in trace["steps"] if s.get("tool_calls")), None
    )
    assert tool_call_step is not None
    assert tool_call_step["tool_calls"][0]["tool"] == "web_search"


@pytest.mark.asyncio
async def test_agent_node_with_prompt_template():
    """Test AgentNode with prompt that references trigger data."""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_chat_model = MagicMock()
    received_prompts = []

    def mock_text_handler(invocation):
        received_prompts.append(invocation.get("prompt", ""))
        return "Mock response", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "data_trigger",
                "key": "test.data",
                "mode": "webhook",
                "event_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                            },
                        },
                    },
                },
            }
        ],
        "nodes": [
            {
                "id": "research",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Research the following topic: ${data_trigger.data.topic}",
                    "tools": [],
                },
                "outputs": {"mode": "text"},
            }
        ],
        "edges": [
            {"source": "data_trigger", "target": "research", "type": "trigger"},
        ],
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Research the following topic: AI Safety"),
            AIMessage(content="Research completed on AI Safety."),
        ]
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def])
        trigger_envelope = {
            "id": "data_trigger",
            "trigger_id": "data_trigger",
            "trigger_key": "test.data",
            "data": {"topic": "AI Safety"},
        }
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    # Check the trace to verify prompt was rendered
    trace = result["_trace_research"]
    assert "AI Safety" in trace["prompt"]


@pytest.mark.asyncio
async def test_agent_node_json_output_mode():
    """Test AgentNode with JSON output mode parses and validates output."""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_chat_model = MagicMock()

    def mock_json_handler(invocation, schema):
        return {"name": "test", "value": 42}, {}

    model_def = ModelDefinition(
        model_id="test-model",
        json_handler=mock_json_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "json_trigger",
                "key": "test.json",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "json_agent",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Extract structured data",
                    "tools": [],
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "value": {"type": "integer"},
                            },
                        }
                    },
                },
            }
        ],
        "edges": [
            {"source": "json_trigger", "target": "json_agent", "type": "trigger"},
        ],
    }

    mock_agent = AsyncMock()
    # Agent returns JSON in markdown code block (common format)
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Extract structured data"),
            AIMessage(content='```json\n{"name": "test", "value": 42}\n```'),
        ]
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def])
        trigger_envelope = {"trigger_key": "test.json"}
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "json_agent" in result
    assert result["json_agent"]["name"] == "test"
    assert result["json_agent"]["value"] == 42


@pytest.mark.asyncio
async def test_agent_node_after_other_node():
    """Test AgentNode receiving output from a previous node."""
    from langchain_core.messages import AIMessage, HumanMessage

    mock_chat_model = MagicMock()

    def mock_text_handler(invocation):
        return "Mock response", {}

    model_def = ModelDefinition(
        model_id="test-model",
        text_handler=mock_text_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    mock_tool = _create_mock_tool_for_agent()

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "chain_trigger",
                "key": "test.chain",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "prepare",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "prepared data"},
            },
            {
                "id": "process_agent",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Process this data: ${prepare}",
                    "tools": [],
                },
                "outputs": {"mode": "text"},
            },
        ],
        "edges": [
            {"source": "chain_trigger", "target": "prepare", "type": "trigger"},
            {"source": "prepare", "target": "process_agent", "type": "default"},
        ],
    }

    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Process this data: prepared data"),
            AIMessage(content="Processed the prepared data successfully."),
        ]
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def], [mock_tool])
        trigger_envelope = {"trigger_key": "test.chain"}
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert result["prepare"] == "prepared data"
    assert "process_agent" in result

    # Verify the prompt was correctly resolved with the previous node's output
    trace = result["_trace_process_agent"]
    assert "prepared data" in trace["prompt"]


@pytest.mark.asyncio
async def test_agent_node_json_output_with_structured_response():
    """Test AgentNode with JSON output mode using structured_response from ToolStrategy."""
    from langchain_core.messages import AIMessage, HumanMessage
    from pydantic import BaseModel

    mock_chat_model = MagicMock()

    def mock_json_handler(invocation, schema):
        return {"email_1": "email 1 summary", "email_2": "email 2 summary"}, {}

    model_def = ModelDefinition(
        model_id="test-model",
        json_handler=mock_json_handler,
        chat_model_factory=lambda: mock_chat_model,
    )

    spec = {
        "version": "2",
        "triggers": [
            {
                "id": "structured_trigger",
                "key": "test.structured",
                "mode": "webhook",
                "event_schema": {"type": "object"},
            }
        ],
        "nodes": [
            {
                "id": "structured_agent",
                "type": "agent",
                "inputs": {
                    "model": "test-model",
                    "prompt": "Summarize my last 2 emails",
                    "tools": [],
                },
                "outputs": {
                    "mode": "json",
                    "schema": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "email_1": {"type": "string", "description": "Summary of first email"},
                                "email_2": {"type": "string", "description": "Summary of second email"},
                            },
                        }
                    },
                },
            }
        ],
        "edges": [
            {"source": "structured_trigger", "target": "structured_agent", "type": "trigger"},
        ],
    }

    # Create a mock Pydantic model to simulate ToolStrategy output
    class MockStructuredOutput(BaseModel):
        email_1: str
        email_2: str

    mock_structured_output = MockStructuredOutput(
        email_1="First email is about a meeting",
        email_2="Second email is about a project update"
    )

    mock_agent = AsyncMock()
    # Agent returns structured_response when using ToolStrategy
    mock_agent.ainvoke.return_value = {
        "messages": [
            HumanMessage(content="Summarize my last 2 emails"),
            AIMessage(content="I've summarized the emails."),
        ],
        "structured_response": mock_structured_output,  # This is set by ToolStrategy
    }

    with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
        compiled = await _compile_agent_workflow(spec, [model_def])
        trigger_envelope = {"trigger_key": "test.structured"}
        result = await compiled.ainvoke(config=None, context=None, trigger=trigger_envelope)

    assert "structured_agent" in result
    # Verify structured output was extracted from Pydantic model
    assert result["structured_agent"]["email_1"] == "First email is about a meeting"
    assert result["structured_agent"]["email_2"] == "Second email is about a project update"

    # Verify trace
    trace = result["_trace_structured_agent"]
    assert trace["status"] == "succeeded"
    assert trace["output"]["email_1"] == "First email is about a meeting"


# =============================================================================
# File Input Resolution Tests
# (Ported from test_llm_file_inputs.py — function now lives in agent_node.py)
# =============================================================================


from datetime import datetime, timezone

from seer.core.files.models import WorkflowFileRef, is_file_ref
from seer.core.nodes.agent_node import _resolve_llm_file_inputs
from seer.core.runtime.context import WorkflowRuntimeContext
from seer.database import User


def _create_file_ref(
    file_id: str = "test-file-123",
    filename: str = "document.pdf",
    mime_type: str = "application/pdf",
    size_bytes: int = 1024,
) -> dict:
    """Create a file reference dict for testing."""
    ref = WorkflowFileRef(
        file_id=file_id,
        storage_path=f"s3://bucket/user/run/{file_id}/{filename}",
        filename=filename,
        mime_type=mime_type,
        size_bytes=size_bytes,
        workflow_run_id="run_123",
        created_at=datetime.now(timezone.utc),
    )
    return ref.to_dict()


def _create_mock_context_with_file_system() -> MagicMock:
    """Create a mock workflow context with file system."""
    mock_user = MagicMock(spec=User)
    mock_user.user_id = "usr_test"

    context = MagicMock(spec=WorkflowRuntimeContext)
    context.user = mock_user
    context.workflow_run_id = "run_test123"

    mock_fs = AsyncMock()
    mock_fs.get_file_content = AsyncMock(return_value=b"file content bytes")
    context.file_system = mock_fs
    context.has_file_system = True

    return context


class TestAgentNodeFileInputResolution:
    """Tests for _resolve_llm_file_inputs ported to agent_node.py."""

    @pytest.mark.asyncio
    async def test_resolve_single_file_ref(self):
        """Single file reference is resolved correctly."""
        context = _create_mock_context_with_file_system()
        file_ref = _create_file_ref(filename="report.pdf", mime_type="application/pdf", size_bytes=2048)

        auxiliary = {"document": file_ref, "other_param": "string value"}
        resolved, file_contents = await _resolve_llm_file_inputs(auxiliary, context)

        assert "_resolved_file" in resolved["document"]
        assert resolved["document"]["_resolved_file"] == "report.pdf"
        assert resolved["other_param"] == "string value"
        assert len(file_contents) == 1
        assert file_contents[0]["filename"] == "report.pdf"
        assert file_contents[0]["content"] == b"file content bytes"

    @pytest.mark.asyncio
    async def test_resolve_list_of_file_refs(self):
        """List of file references is resolved correctly."""
        context = _create_mock_context_with_file_system()
        file_ref1 = _create_file_ref(filename="image1.png", mime_type="image/png")
        file_ref2 = _create_file_ref(filename="image2.png", mime_type="image/png")

        auxiliary = {"attachments": [file_ref1, file_ref2]}
        resolved, file_contents = await _resolve_llm_file_inputs(auxiliary, context)

        assert len(resolved["attachments"]) == 2
        assert resolved["attachments"][0]["_resolved_file"] == "image1.png"
        assert len(file_contents) == 2

    @pytest.mark.asyncio
    async def test_no_file_refs_returns_original(self):
        """When no file refs, original inputs returned unchanged."""
        context = _create_mock_context_with_file_system()
        auxiliary = {"param1": "value1", "param2": 123}

        resolved, file_contents = await _resolve_llm_file_inputs(auxiliary, context)

        assert resolved == auxiliary
        assert file_contents == []

    @pytest.mark.asyncio
    async def test_no_context_returns_original(self):
        """Without context, original inputs returned unchanged."""
        file_ref = _create_file_ref()
        auxiliary = {"document": file_ref}

        resolved, file_contents = await _resolve_llm_file_inputs(auxiliary, None)

        assert resolved == auxiliary
        assert file_contents == []

    @pytest.mark.asyncio
    async def test_context_without_file_system(self):
        """Context without file system returns original inputs."""
        mock_user = MagicMock(spec=User)
        context = WorkflowRuntimeContext(user=mock_user)

        file_ref = _create_file_ref()
        auxiliary = {"document": file_ref}

        with patch.object(WorkflowRuntimeContext, "has_file_system", new_callable=lambda: property(lambda self: False)):
            resolved, file_contents = await _resolve_llm_file_inputs(auxiliary, context)

        assert resolved == auxiliary
        assert file_contents == []

    @pytest.mark.asyncio
    async def test_file_contents_added_to_trace_inputs(self):
        """File contents info is added to trace inputs when files are present."""
        from langchain_core.messages import AIMessage, HumanMessage

        context = _create_mock_context_with_file_system()
        file_ref = _create_file_ref(filename="data.pdf", mime_type="application/pdf")

        mock_chat_model = MagicMock()
        model_def = ModelDefinition(
            model_id="test-model",
            text_handler=lambda inv: ("result", {}),
            chat_model_factory=lambda: mock_chat_model,
        )

        spec = {
            "version": "2",
            "triggers": [
                {
                    "id": "file_trigger",
                    "key": "test.file",
                    "mode": "webhook",
                    "event_schema": {"type": "object"},
                }
            ],
            "nodes": [
                {
                    "id": "file_agent",
                    "type": "agent",
                    "inputs": {
                        "model": "test-model",
                        "prompt": "Analyze the document",
                        "tools": [],
                    },
                    "outputs": {"mode": "text"},
                }
            ],
            "edges": [
                {"source": "file_trigger", "target": "file_agent", "type": "trigger"},
            ],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                HumanMessage(content="Analyze the document"),
                AIMessage(content="Document analyzed."),
            ]
        }

        with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
            compiled = await _compile_agent_workflow(spec, [model_def])
            trigger_envelope = {"trigger_key": "test.file"}
            # Inject file_ref into state manually via context to simulate file input
            result = await compiled.ainvoke(config=None, context=context, trigger=trigger_envelope)

        assert "file_agent" in result
        assert result["file_agent"] == "Document analyzed."
