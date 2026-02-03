"""
Unit tests for core.runtime.nodes module.

Tests NodeRuntime class, node execution, and trace key generation.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from datetime import datetime, timezone
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

        with patch.object(node_runtime, "_run_node", return_value={"result": "ok"}):
            runner = node_runtime.build_runner(node)

        assert runner is not None
        assert runner.name == "node:tool_node_1"

    def test_build_runner_runnable_name(self, node_runtime):
        """Test runner has correct name from node ID."""
        from seer.core.schema.models import LLMNode

        node = LLMNode(id="llm_node_abc", inputs={"model": "gpt-4", "prompt": "test"})
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


@pytest.mark.unit
class TestRunNodeDispatch:
    """Tests for NodeRuntime._run_node dispatch logic."""

    def test_run_node_tool_dispatch(self, node_runtime):
        """Test _run_node dispatches ToolNode correctly."""
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="tool_1", tool="test.tool")
        state = {}

        with patch.object(node_runtime, "_run_tool", return_value={"result": "ok"}) as mock_run:
            result = node_runtime._run_node(node, state, {}, locals_ctx=None, context=None)

        mock_run.assert_called_once()
        assert result == {"result": "ok"}

    def test_run_node_llm_dispatch(self, node_runtime):
        """Test _run_node dispatches LLMNode correctly."""
        from seer.core.schema.models import LLMNode

        node = LLMNode(id="llm_1", inputs={"model": "gpt-4", "prompt": "test"})
        state = {}

        with patch.object(node_runtime, "_check_llm_credit_limit_sync"):
            with patch.object(node_runtime, "_run_llm", return_value={"output": "response"}) as mock_run:
                result = node_runtime._run_node(node, state, {}, locals_ctx=None, context=None)

        mock_run.assert_called_once()
        assert result == {"output": "response"}

    def test_run_node_if_dispatch(self, node_runtime):
        """Test _run_node dispatches IfNode correctly."""
        from seer.core.schema.models import IfNode

        # IfNode uses edges with conditional types for branches, not then_nodes/else_nodes
        node = IfNode(id="if_1", condition="${input.value > 10}")
        state = {}

        with patch.object(node_runtime, "_run_if", return_value={}) as mock_run:
            node_runtime._run_node(node, state, {}, locals_ctx=None, context=None)

        mock_run.assert_called_once()

    def test_run_node_for_each_dispatch(self, node_runtime):
        """Test _run_node dispatches ForEachNode correctly."""
        from seer.core.schema.models import ForEachNode

        # ForEachNode uses edges with loop_body type for body, not body_nodes
        node = ForEachNode(id="for_1", items="${input.items}", item_var="item")
        state = {}

        with patch.object(node_runtime, "_run_for_each", return_value={}) as mock_run:
            node_runtime._run_node(node, state, {}, locals_ctx=None, context=None)

        mock_run.assert_called_once()

    def test_run_node_unsupported_type_raises(self, node_runtime):
        """Test _run_node raises for unsupported node type."""
        from seer.core.errors import ExecutionError

        # Create a mock node with unsupported type
        mock_node = MagicMock()
        mock_node.id = "unknown_1"
        mock_node.type = "unknown"

        with pytest.raises(ExecutionError, match="Unsupported node type"):
            node_runtime._run_node(mock_node, {}, {}, locals_ctx=None, context=None)


# =============================================================================
# Run Node Async Dispatch Tests
# =============================================================================


@pytest.mark.unit
class TestRunNodeAsyncDispatch:
    """Tests for NodeRuntime._run_node_async dispatch logic."""

    @pytest.mark.asyncio
    async def test_run_node_async_tool_dispatch(self, node_runtime):
        """Test _run_node_async dispatches ToolNode correctly."""
        from seer.core.schema.models import ToolNode

        node = ToolNode(id="tool_async_1", tool="test.async_tool")
        state = {}

        with patch.object(node_runtime, "_run_tool_async", new_callable=AsyncMock, return_value={"async_result": "ok"}) as mock_run:
            result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        mock_run.assert_called_once()
        assert result == {"async_result": "ok"}

    @pytest.mark.asyncio
    async def test_run_node_async_llm_dispatch(self, node_runtime):
        """Test _run_node_async dispatches LLMNode with credit check."""
        from seer.core.schema.models import LLMNode

        node = LLMNode(id="llm_async_1", inputs={"model": "gpt-4", "prompt": "test"})
        state = {}

        with patch.object(node_runtime, "_check_llm_credit_limit_async", new_callable=AsyncMock) as mock_credit:
            with patch.object(node_runtime, "_run_llm", return_value={"llm_output": "response"}) as mock_run:
                result = await node_runtime._run_node_async(node, state, {}, locals_ctx=None, context=None)

        mock_credit.assert_called_once()
        mock_run.assert_called_once()
        assert result == {"llm_output": "response"}


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

    def test_check_llm_credit_limit_sync_no_context(self, node_runtime):
        """Test sync credit check skips when no context."""
        node_runtime._current_context = None

        # Should not raise
        node_runtime._check_llm_credit_limit_sync()


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
