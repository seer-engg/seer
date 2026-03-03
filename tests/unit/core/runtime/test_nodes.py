"""
Unit tests for core.runtime.nodes module.

Tests NodeRuntime class, node execution, and trace key generation.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_services():
    """Create mock RuntimeServices."""
    from seer.core.runtime.nodes import RuntimeServices

    mock_schema_registry = MagicMock()
    mock_tool_registry = MagicMock()
    mock_model_registry = MagicMock()
    mock_type_env = MagicMock()
    mock_type_env.as_dict.return_value = {}

    return RuntimeServices(
        schema_registry=mock_schema_registry,
        tool_registry=mock_tool_registry,
        model_registry=mock_model_registry,
        type_env=mock_type_env,
        mcp_client_registry=None,
    )


@pytest.fixture
def node_runtime(mock_services):
    """Create a NodeRuntime instance."""
    from seer.core.runtime.nodes import NodeRuntime
    return NodeRuntime(mock_services)


# =============================================================================
# RuntimeServices Tests
# =============================================================================


@pytest.mark.unit
class TestRuntimeServices:
    """Tests for RuntimeServices dataclass."""

    def test_runtime_services_creation(self):
        """Test creating RuntimeServices with all fields."""
        from seer.core.runtime.nodes import RuntimeServices

        schema_registry = MagicMock()
        tool_registry = MagicMock()
        model_registry = MagicMock()
        type_env = MagicMock()
        mcp_registry = MagicMock()

        services = RuntimeServices(
            schema_registry=schema_registry,
            tool_registry=tool_registry,
            model_registry=model_registry,
            type_env=type_env,
            mcp_client_registry=mcp_registry,
        )

        assert services.schema_registry == schema_registry
        assert services.tool_registry == tool_registry
        assert services.model_registry == model_registry
        assert services.type_env == type_env
        assert services.mcp_client_registry == mcp_registry

    def test_runtime_services_optional_mcp(self):
        """Test RuntimeServices without MCP registry."""
        from seer.core.runtime.nodes import RuntimeServices

        services = RuntimeServices(
            schema_registry=MagicMock(),
            tool_registry=MagicMock(),
            model_registry=MagicMock(),
            type_env=MagicMock(),
        )

        assert services.mcp_client_registry is None


# =============================================================================
# NodeRuntime Build Runner Tests
# =============================================================================


@pytest.mark.unit
class TestNodeRuntimeBuildRunner:
    """Tests for NodeRuntime.build_runner method."""

    def test_build_runner_creates_runnable(self, node_runtime):
        """Test build_runner creates a RunnableCallable."""
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="tool_node_1", tool="test.tool")
        runner = node_runtime.build_runner(node)

        assert runner is not None
        assert runner.name == "node:tool_node_1"

    def test_build_runner_runnable_name(self, node_runtime):
        """Test runner has correct name from node ID."""
        from seer.core.schema.models import AgentNode

        node = AgentNode(id="llm_node_abc", inputs={"model": "gpt-4", "prompt": "test"})
        runner = node_runtime.build_runner(node)

        assert "llm_node_abc" in runner.name


# =============================================================================
# Get Trace Key Tests
# =============================================================================


@pytest.mark.unit
class TestGetTraceKey:
    """Tests for NodeRuntime._get_trace_key method."""

    def test_get_trace_key_simple_node(self, node_runtime):
        """Test trace key for node not in a loop."""
        state = {}

        result = node_runtime._get_trace_key("node_123", state)

        assert result == "_trace_node_123"

    def test_get_trace_key_node_in_loop(self, node_runtime):
        """Test trace key for node inside a loop includes iteration."""
        node_runtime.set_loop_body_map({"child_node": "loop_1"})
        state = {"_loop_loop_1": {"current_index": 5}}

        result = node_runtime._get_trace_key("child_node", state)

        assert result == "_trace_child_node_iter_5"

    def test_get_trace_key_loop_no_state(self, node_runtime):
        """Test trace key for loop node when loop state missing."""
        node_runtime.set_loop_body_map({"child_node": "loop_1"})
        state = {}  # No loop state

        result = node_runtime._get_trace_key("child_node", state)

        assert result == "_trace_child_node"

    def test_get_trace_key_loop_invalid_state(self, node_runtime):
        """Test trace key when loop state is not a dict."""
        node_runtime.set_loop_body_map({"child_node": "loop_1"})
        state = {"_loop_loop_1": "not_a_dict"}

        result = node_runtime._get_trace_key("child_node", state)

        assert result == "_trace_child_node"


# =============================================================================
# Bind Trigger Tests
# =============================================================================


@pytest.mark.unit
class TestBindTrigger:
    """Tests for NodeRuntime.bind_trigger method."""

    def test_bind_trigger_with_data(self, node_runtime):
        """Test binding trigger data."""
        trigger = {"data": {"message": "test"}, "provider": "webhook"}

        node_runtime.bind_trigger(trigger)

        assert node_runtime._current_trigger == trigger

    def test_bind_trigger_none(self, node_runtime):
        """Test binding None trigger clears current trigger."""
        node_runtime._current_trigger = {"old": "data"}

        node_runtime.bind_trigger(None)

        assert node_runtime._current_trigger is None


# =============================================================================
# Bind Context Tests
# =============================================================================


@pytest.mark.unit
class TestBindContext:
    """Tests for NodeRuntime.bind_context method."""

    def test_bind_context(self, node_runtime):
        """Test binding runtime context."""
        mock_context = MagicMock()

        node_runtime.bind_context(mock_context)

        assert node_runtime._current_context == mock_context

    def test_bind_context_none(self, node_runtime):
        """Test binding None context."""
        node_runtime._current_context = MagicMock()

        node_runtime.bind_context(None)

        assert node_runtime._current_context is None


# =============================================================================
# Set Loop Body Map Tests
# =============================================================================


@pytest.mark.unit
class TestSetLoopBodyMap:
    """Tests for NodeRuntime.set_loop_body_map method."""

    def test_set_loop_body_map(self, node_runtime):
        """Test setting loop body map."""
        loop_map = {"node_a": "loop_1", "node_b": "loop_1", "node_c": "loop_2"}

        node_runtime.set_loop_body_map(loop_map)

        assert node_runtime._loop_body_map == loop_map

    def test_set_loop_body_map_empty(self, node_runtime):
        """Test setting empty loop body map."""
        node_runtime._loop_body_map = {"existing": "data"}

        node_runtime.set_loop_body_map({})

        assert node_runtime._loop_body_map == {}


# =============================================================================
# Run Node Dispatch Tests
# =============================================================================


# =============================================================================
# Run Node Async Dispatch Tests
# =============================================================================


@pytest.mark.unit
class TestRunNodeAsyncDispatch:
    """Tests for NodeRuntime._run_node_async dispatch logic using registry."""

    @pytest.mark.asyncio
    async def test_run_node_async_tool_dispatch(self, node_runtime):
        """Test _run_node_async dispatches ToolNode through registry."""
        from seer.core.schema.models import ToolNode
        from seer.core.nodes.registry import node_type_registry

        node = ToolNode(id="tool_async_1", tool="test.async_tool")
        state = {}

        # Get the registered tool node type and mock its execute_async
        tool_node_type = node_type_registry.get("tool")
        with patch.object(tool_node_type, "execute_async", new_callable=AsyncMock, return_value={"async_result": "ok"}) as mock_exec:
            result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        mock_exec.assert_called_once()
        assert result == {"async_result": "ok"}

    @pytest.mark.asyncio
    async def test_run_node_async_agent_dispatch(self, node_runtime):
        """Test _run_node_async dispatches AgentNode through registry."""
        from seer.core.schema.models import AgentNode
        from seer.core.nodes.registry import node_type_registry

        node = AgentNode(id="agent_async_1", inputs={"model": "gpt-4", "prompt": "test"})
        state = {}

        # Get the registered agent node type and mock its execute_async
        agent_node_type = node_type_registry.get("agent")
        with patch.object(agent_node_type, "execute_async", new_callable=AsyncMock, return_value={"agent_output": "response"}) as mock_exec:
            result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        mock_exec.assert_called_once()
        assert result == {"agent_output": "response"}


# =============================================================================
# Credit Limit Check Tests
# =============================================================================


@pytest.mark.unit
class TestCreditLimitCheck:
    """Tests for credit limit checking methods."""

    @pytest.mark.asyncio
    async def test_check_llm_credit_limit_async_no_context(self, node_runtime):
        """Test async credit check skips when no context."""
        node_runtime._current_context = None

        # Should not raise
        await node_runtime._check_llm_credit_limit_async()

    @pytest.mark.asyncio
    async def test_check_llm_credit_limit_async_no_user(self, node_runtime):
        """Test async credit check skips when no user in context."""
        mock_context = MagicMock()
        mock_context.user = None
        node_runtime._current_context = mock_context

        # Should not raise
        await node_runtime._check_llm_credit_limit_async()

    @pytest.mark.asyncio
    async def test_check_llm_credit_limit_async_success(self, node_runtime):
        """Test async credit check passes when under limit."""
        mock_context = MagicMock()
        mock_context.user = MagicMock()
        node_runtime._current_context = mock_context

        # check_credit_limit is imported inside the function from seer.observability.credit_gate
        with patch("seer.observability.credit_gate.check_credit_limit", new_callable=AsyncMock):
            # Should not raise
            await node_runtime._check_llm_credit_limit_async()


# =============================================================================
# Track LLM Usage Tests
# =============================================================================


@pytest.mark.unit
class TestTrackLlmUsage:
    """Tests for LLM usage tracking."""

    def test_track_llm_usage_no_context(self, node_runtime):
        """Test usage tracking skips when no context."""
        node_runtime._current_context = None

        # Should not raise, just log warning
        with patch("seer.core.runtime.nodes.logger") as mock_logger:
            node_runtime._track_llm_usage_async({"model": "gpt-4", "input_tokens": 100})

        mock_logger.warning.assert_called()

    def test_track_llm_usage_no_user(self, node_runtime):
        """Test usage tracking skips when no user."""
        mock_context = MagicMock()
        mock_context.user = None
        node_runtime._current_context = mock_context

        with patch("seer.core.runtime.nodes.logger") as mock_logger:
            node_runtime._track_llm_usage_async({"model": "gpt-4", "input_tokens": 100})

        mock_logger.warning.assert_called()


# =============================================================================
# Error Trace Tests
# =============================================================================


@pytest.mark.unit
class TestErrorTraceCapture:
    """Tests for error trace capture when node execution fails."""

    def test_write_error_trace_creates_proper_structure(self):
        """Test write_error_trace creates proper trace structure."""
        from seer.core.nodes.base import write_error_trace

        state = {}
        inputs = {"param1": "value1", "param2": 42}
        exc = ValueError("Something went wrong")

        result = write_error_trace(
            node_id="test_node",
            node_type="tool",
            inputs=inputs,
            exc=exc,
            state=state,
            loop_body_map={},
            nested_loop_parents={},
        )

        assert "_trace_test_node" in result
        trace = result["_trace_test_node"]
        assert trace["node_id"] == "test_node"
        assert trace["node_type"] == "tool"
        assert trace["inputs"] == inputs
        assert trace["error"]["type"] == "ValueError"
        assert trace["error"]["message"] == "Something went wrong"
        assert trace["status"] == "failed"
        assert "timestamp" in trace

    @pytest.mark.asyncio
    async def test_tool_node_failure_writes_error_trace(self, node_runtime, mock_services):
        """Test that tool node failures include error trace in exception."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="failing_tool", tool="test.fail_tool", inputs={"key": "value"})
        state = {}

        # Setup mock tool that raises an exception
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=RuntimeError("Tool execution failed"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError) as exc_info:
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify error trace is attached
        assert exc_info.value.trace_data is not None
        trace_key = "_trace_failing_tool"
        assert trace_key in exc_info.value.trace_data
        trace = exc_info.value.trace_data[trace_key]
        assert trace["node_id"] == "failing_tool"
        assert trace["node_type"] == "tool"
        assert trace["status"] == "failed"
        assert trace["error"]["type"] == "RuntimeError"
        assert "Tool execution failed" in trace["error"]["message"]

    @pytest.mark.asyncio
    async def test_tool_node_async_failure_writes_error_trace(self, node_runtime, mock_services):
        """Test that async tool node failures include error trace in exception."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="async_failing_tool", tool="test.async_fail", inputs={"data": 123})
        state = {}

        # Setup mock tool with async handler that raises
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=ConnectionError("Connection lost"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError) as exc_info:
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify error trace is attached
        assert exc_info.value.trace_data is not None
        trace_key = "_trace_async_failing_tool"
        assert trace_key in exc_info.value.trace_data
        trace = exc_info.value.trace_data[trace_key]
        assert trace["node_id"] == "async_failing_tool"
        assert trace["node_type"] == "tool"
        assert trace["status"] == "failed"
        assert trace["error"]["type"] == "ConnectionError"

    @pytest.mark.asyncio
    async def test_mcp_node_failure_writes_error_trace(self, mock_services):
        """Test that MCP node failures include error trace in exception."""
        from seer.core.errors import ExecutionError
        from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
        from seer.core.schema.models import MCPNode

        # Create NodeRuntime with MCP registry
        mock_mcp_registry = MagicMock()
        mock_mcp_registry.invoke_tool = AsyncMock(side_effect=TimeoutError("MCP server timeout"))

        services = RuntimeServices(
            schema_registry=mock_services.schema_registry,
            tool_registry=mock_services.tool_registry,
            model_registry=mock_services.model_registry,
            type_env=mock_services.type_env,
            mcp_client_registry=mock_mcp_registry,
        )
        node_runtime = NodeRuntime(services)

        node = MCPNode(
            id="mcp_fail",
            tool="mcp_tool",
            server="http://test-server",
            server_type="http",
            inputs={"query": "test"},
        )
        state = {}

        with pytest.raises(ExecutionError) as exc_info:
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify error trace is attached
        assert exc_info.value.trace_data is not None
        trace_key = "_trace_mcp_fail"
        assert trace_key in exc_info.value.trace_data
        trace = exc_info.value.trace_data[trace_key]
        assert trace["node_id"] == "mcp_fail"
        assert trace["node_type"] == "mcp"
        assert trace["status"] == "failed"

    @pytest.mark.asyncio
    async def test_tool_node_success_has_status_succeeded(self, node_runtime, mock_services):
        """Test that successful tool execution includes status: succeeded."""
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="success_tool", tool="test.success", inputs={"x": 1})
        state = {}

        # Setup mock tool that succeeds
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(return_value={"result": "success"})
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify success trace has status
        trace_key = "_trace_success_tool"
        assert trace_key in result
        assert result[trace_key]["status"] == "succeeded"
        assert "error" not in result[trace_key]

    @pytest.mark.asyncio
    async def test_tool_node_async_success_has_status_succeeded(self, node_runtime, mock_services):
        """Test that async successful tool execution includes status: succeeded."""
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="async_success", tool="test.async_ok", inputs={"y": 2})
        state = {}

        # Setup mock tool with async handler that succeeds
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(return_value={"async_result": "done"})
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify success trace has status
        trace_key = "_trace_async_success"
        assert trace_key in result
        assert result[trace_key]["status"] == "succeeded"

    @pytest.mark.asyncio
    async def test_agent_node_failure_writes_error_trace(self, node_runtime, mock_services):
        """Test that agent node failures include error trace in exception."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import AgentNode, OutputContract, OutputMode

        node = AgentNode(
            id="llm_fail",
            inputs={"model": "gpt-4", "prompt": "Hello ${user}"},
            outputs=OutputContract(mode=OutputMode.text),
        )
        state = {"user": "Alice"}

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("API rate limit exceeded"))
        with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
            with pytest.raises(ExecutionError) as exc_info:
                await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify error trace is attached
        assert exc_info.value.trace_data is not None
        trace_key = "_trace_llm_fail"
        assert trace_key in exc_info.value.trace_data
        trace = exc_info.value.trace_data[trace_key]
        assert trace["node_id"] == "llm_fail"
        assert trace["node_type"] == "agent"
        assert trace["status"] == "failed"
        assert trace["error"]["type"] == "RuntimeError"
        assert "API rate limit exceeded" in trace["error"]["message"]

    @pytest.mark.asyncio
    async def test_llm_node_success_has_status_succeeded(self, node_runtime, mock_services):
        """Test that successful agent execution includes status: succeeded."""
        from langchain_core.messages import AIMessage
        from seer.core.schema.models import AgentNode, OutputContract, OutputMode

        node = AgentNode(
            id="llm_success",
            inputs={"model": "gpt-4", "prompt": "Hello"},
            outputs=OutputContract(mode=OutputMode.text),
        )
        state = {}

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello response")]})
        with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
            result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify success trace has status
        trace_key = "_trace_llm_success"
        assert trace_key in result
        assert result[trace_key]["status"] == "succeeded"
        assert "error" not in result[trace_key]

    @pytest.mark.asyncio
    async def test_error_trace_in_loop_includes_iteration(self, node_runtime, mock_services):
        """Test that error trace in loop includes iteration index."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="loop_tool", tool="test.loop_fail", inputs={})

        # Set up loop context
        node_runtime.set_loop_body_map({"loop_tool": "my_loop"})
        state = {"_loop_my_loop": {"current_index": 2, "items": [1, 2, 3]}}

        # Setup mock tool that raises
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=ValueError("Failed at index 2"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError) as exc_info:
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # Verify trace key includes iteration
        assert exc_info.value.trace_data is not None
        trace_key = "_trace_loop_tool_iter_2"
        assert trace_key in exc_info.value.trace_data


# =============================================================================
# Error Trace State Persistence Tests
# =============================================================================


@pytest.mark.unit
class TestErrorTraceStatePersistence:
    """Test that error traces are persisted to state dict before raising exceptions.

    This ensures failed nodes appear in workflow history API responses.
    """

    @pytest.mark.asyncio
    async def test_tool_failure_updates_state_with_error_trace(self, node_runtime, mock_services):
        """Verify tool node failures write error trace to state before raising."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(
            id='failing_tool',
            tool='mock_tool',
            inputs={'param': 'value'},
        )
        state = {}

        # Setup mock tool that raises
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=KeyError("missing_key"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError) as exc_info:
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # VERIFY: State dict was updated with error trace
        assert '_trace_failing_tool' in state, "Error trace key missing from state"
        trace = state['_trace_failing_tool']

        assert trace['node_id'] == 'failing_tool'
        assert trace['node_type'] == 'tool'
        assert trace['status'] == 'failed'
        assert 'error' in trace
        assert trace['error']['type'] == 'KeyError'
        assert 'missing_key' in trace['error']['message']
        assert 'timestamp' in trace

        # VERIFY: Exception also contains trace_data (backward compatibility)
        assert exc_info.value.trace_data is not None
        assert '_trace_failing_tool' in exc_info.value.trace_data

    @pytest.mark.asyncio
    async def test_tool_async_failure_updates_state_with_error_trace(self, node_runtime, mock_services):
        """Verify async tool node failures write error trace to state before raising."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(
            id='failing_async_tool',
            tool='mock_async_tool',
            inputs={'param': 'value'},
        )
        state = {}

        # Setup mock tool that raises
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=ValueError("async error"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError):
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # VERIFY: State dict was updated with error trace
        assert '_trace_failing_async_tool' in state
        trace = state['_trace_failing_async_tool']

        assert trace['status'] == 'failed'
        assert 'error' in trace
        assert trace['error']['type'] == 'ValueError'
        assert 'async error' in trace['error']['message']

    @pytest.mark.asyncio
    async def test_agent_failure_updates_state_with_error_trace(self, node_runtime, mock_services):
        """Verify agent node failures write error trace to state before raising."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import AgentNode, OutputContract, OutputMode

        node = AgentNode(
            id='failing_llm',
            inputs={'prompt': 'test prompt', 'model': 'gpt-4'},
            outputs=OutputContract(mode=OutputMode.text),
        )
        state = {}

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=RuntimeError("Model unavailable"))
        with patch("seer.core.nodes.agent_node.create_agent", return_value=mock_agent):
            with pytest.raises(ExecutionError):
                await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # VERIFY: State dict was updated with error trace
        assert '_trace_failing_llm' in state
        trace = state['_trace_failing_llm']

        assert trace['node_id'] == 'failing_llm'
        assert trace['node_type'] == 'agent'
        assert trace['status'] == 'failed'
        assert 'error' in trace
        assert trace['error']['type'] == 'RuntimeError'
        assert 'Model unavailable' in trace['error']['message']
        assert 'timestamp' in trace

    @pytest.mark.asyncio
    async def test_mcp_failure_updates_state_with_error_trace(self):
        """Verify MCP node failures write error trace to state before raising."""
        from seer.core.errors import ExecutionError
        from seer.core.runtime.nodes import NodeRuntime, RuntimeServices
        from seer.core.schema.models import MCPNode

        # Create services with MCP registry included
        mock_mcp_registry = MagicMock()
        mock_mcp_registry.invoke_tool = AsyncMock(side_effect=ConnectionError("MCP server unreachable"))
        mock_services = RuntimeServices(
            schema_registry=MagicMock(),
            tool_registry=MagicMock(),
            model_registry=MagicMock(),
            type_env=MagicMock(),
            mcp_client_registry=mock_mcp_registry,
        )
        node_runtime = NodeRuntime(mock_services)

        node = MCPNode(
            id='failing_mcp',
            tool='mcp_mock_tool',
            inputs={'param': 'value'},
            server='test_mcp_server',
            server_type='stdio',
        )
        state = {}

        with pytest.raises(ExecutionError):
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # VERIFY: State dict was updated with error trace
        assert '_trace_failing_mcp' in state
        trace = state['_trace_failing_mcp']

        assert trace['node_id'] == 'failing_mcp'
        assert trace['node_type'] == 'mcp'
        assert trace['status'] == 'failed'
        assert 'error' in trace
        # MCP node wraps errors in ExecutionError
        assert trace['error']['type'] == 'ExecutionError'
        assert 'MCP' in trace['error']['message'] and 'unreachable' in trace['error']['message']

    @pytest.mark.asyncio
    async def test_error_trace_state_persistence_in_loop(self, node_runtime, mock_services):
        """Verify error traces in loops are persisted with iteration index."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        node = ToolNode(
            id='loop_failing_tool',
            tool='test.loop_fail',
            inputs={},
        )

        # Set up loop context for iteration 3
        node_runtime.set_loop_body_map({"loop_failing_tool": "my_loop"})
        state = {"_loop_my_loop": {"current_index": 3, "items": [1, 2, 3, 4, 5]}}

        # Setup mock tool that raises
        mock_tool_def = MagicMock()
        mock_tool_def.async_handler = AsyncMock(side_effect=ValueError("Failed at index 3"))
        mock_tool_def.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def

        with pytest.raises(ExecutionError):
            await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        # VERIFY: State dict was updated with error trace including iteration
        trace_key = "_trace_loop_failing_tool_iter_3"
        assert trace_key in state, f"Expected {trace_key} in state keys: {list(state.keys())}"

        trace = state[trace_key]
        assert trace['status'] == 'failed'
        assert 'error' in trace
        assert '3' in trace['error']['message'] or 'index 3' in trace['error']['message']

    @pytest.mark.asyncio
    async def test_multiple_failures_all_persisted_to_state(self, node_runtime, mock_services):
        """Verify multiple failed nodes all persist their traces to state."""
        from seer.core.errors import ExecutionError
        from seer.core.schema.models import ToolNode

        state = {}

        # First failure
        node1 = ToolNode(id='fail_1', tool='tool_1', inputs={})
        mock_tool_def_1 = MagicMock()
        mock_tool_def_1.async_handler = AsyncMock(side_effect=KeyError("error_1"))
        mock_tool_def_1.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def_1

        with pytest.raises(ExecutionError):
            await node_runtime._run_node_async(node1, state, {}, locals_ctx=None, context=None)

        # Second failure
        node2 = ToolNode(id='fail_2', tool='tool_2', inputs={})
        mock_tool_def_2 = MagicMock()
        mock_tool_def_2.async_handler = AsyncMock(side_effect=ValueError("error_2"))
        mock_tool_def_2.input_schema = {}
        mock_services.tool_registry.get.return_value = mock_tool_def_2

        with pytest.raises(ExecutionError):
            await node_runtime._run_node_async(node2, state, {}, locals_ctx=None, context=None)

        # VERIFY: Both error traces are in state
        assert '_trace_fail_1' in state
        assert '_trace_fail_2' in state
        assert state['_trace_fail_1']['status'] == 'failed'
        assert state['_trace_fail_2']['status'] == 'failed'
        assert 'error_1' in state['_trace_fail_1']['error']['message']
        assert 'error_2' in state['_trace_fail_2']['error']['message']
