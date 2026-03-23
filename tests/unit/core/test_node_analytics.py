"""
Unit tests for per-node analytics: get_analytics_properties() and PostHog capture
in NodeRuntime._run_node_async().
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seer.core.nodes.base import NodeExecutionContext
from seer.core.nodes.tool_node import ToolNodeType
from seer.core.nodes.agent_node import AgentNodeType
from seer.core.schema.models import ToolNode, AgentNode

pytestmark = pytest.mark.unit


# =============================================================================
# Helpers
# =============================================================================

def _make_ctx(user_email: str = "test@example.com", workflow_run_id: str = "run_1") -> NodeExecutionContext:
    runtime_context = MagicMock()
    runtime_context.user.email = user_email
    runtime_context.workflow_run_id = workflow_run_id
    return NodeExecutionContext(
        state={},
        config={},
        locals_ctx=None,
        runtime_context=runtime_context,
    )


# =============================================================================
# get_analytics_properties() per node type
# =============================================================================

class TestToolNodeAnalyticsProperties:
    def test_returns_tool_name(self):
        node = ToolNode(id="n1", tool="gmail_send_email")
        ctx = _make_ctx()
        props = ToolNodeType().get_analytics_properties(node, ctx)
        assert props["tool_name"] == "gmail_send_email"

    def test_returns_connection_id(self):
        node = ToolNode(id="n1", tool="slack_send_message", connection_id=42)
        ctx = _make_ctx()
        props = ToolNodeType().get_analytics_properties(node, ctx)
        assert props["connection_id"] == 42

    def test_connection_id_none_when_absent(self):
        node = ToolNode(id="n1", tool="some_tool")
        ctx = _make_ctx()
        props = ToolNodeType().get_analytics_properties(node, ctx)
        assert props["connection_id"] is None



class TestAgentNodeAnalyticsProperties:
    def test_returns_model_as_agent_type(self):
        node = AgentNode(
            id="n1",
            inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Do something", "tools": []},
        )
        ctx = _make_ctx()
        props = AgentNodeType().get_analytics_properties(node, ctx)
        assert props["agent_type"] == "qwen/qwen3-235b-a22b-2507"

    def test_falls_back_when_model_key_absent_from_inputs(self):
        """get_analytics_properties returns 'unknown' when inputs.get('model') is None."""
        # AgentNode requires model in inputs for validation, but we can test the method
        # directly with a mock node that has an empty inputs dict.
        node = MagicMock()
        node.inputs = {}  # no 'model' key
        ctx = _make_ctx()
        props = AgentNodeType().get_analytics_properties(node, ctx)
        assert props["agent_type"] == "unknown"


# =============================================================================
# NodeRuntime._run_node_async() fires capture_workflow_event
# =============================================================================

@pytest.fixture
def mock_tool_node_impl():
    """Tool node type impl with a successful execute_async."""
    impl = MagicMock()
    impl.execute_async = AsyncMock(return_value={"n1": "result"})
    impl.get_analytics_properties = MagicMock(return_value={"tool_name": "test_tool", "connection_id": None})
    return impl


@pytest.fixture
def mock_failing_node_impl():
    """Node type impl whose execute_async always raises."""
    impl = MagicMock()
    impl.execute_async = AsyncMock(side_effect=RuntimeError("boom"))
    impl.get_analytics_properties = MagicMock(return_value={"tool_name": "bad_tool", "connection_id": None})
    return impl


@pytest.mark.asyncio
async def test_run_node_fires_posthog_on_success(mock_tool_node_impl):
    """capture_workflow_event is called after successful node execution."""
    from seer.core.runtime.nodes import NodeRuntime

    services = MagicMock()
    services.type_env.as_dict.return_value = {}
    runtime = NodeRuntime(services)

    node = MagicMock()
    node.id = "n1"
    node.type = "tool"

    ctx_runtime = MagicMock()
    ctx_runtime.user.email = "user@example.com"
    ctx_runtime.workflow_run_id = "run_42"

    with (
        patch("seer.core.nodes.registry.node_type_registry") as mock_registry,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
        patch("seer.analytics.workflow_tracking.config") as mock_cfg,
    ):
        mock_registry.get.return_value = mock_tool_node_impl
        mock_cfg.is_posthog_configured = True

        result = await runtime._run_node_async(
            node,
            state={},
            config={},
            locals_ctx=None,
            context=ctx_runtime,
        )

        assert result == {"n1": "result"}
        mock_capture.assert_called_once()
        call_kwargs = mock_capture.call_args
        assert call_kwargs.kwargs["event"] == "workflow_node_executed"
        assert call_kwargs.kwargs["distinct_id"] == "user@example.com"
        props = call_kwargs.kwargs["properties"]
        assert props["success"] is True
        assert props["node_id"] == "n1"
        assert props["node_type"] == "tool"
        assert props["tool_name"] == "test_tool"
        assert props["error"] is None


@pytest.mark.asyncio
async def test_run_node_fires_posthog_on_failure(mock_failing_node_impl):
    """capture_workflow_event fires even when node execution raises."""
    from seer.core.runtime.nodes import NodeRuntime

    services = MagicMock()
    services.type_env.as_dict.return_value = {}
    runtime = NodeRuntime(services)

    node = MagicMock()
    node.id = "n2"
    node.type = "tool"

    ctx_runtime = MagicMock()
    ctx_runtime.user.email = "user@example.com"
    ctx_runtime.workflow_run_id = "run_99"

    with (
        patch("seer.core.nodes.registry.node_type_registry") as mock_registry,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
        patch("seer.analytics.workflow_tracking.config") as mock_cfg,
    ):
        mock_registry.get.return_value = mock_failing_node_impl
        mock_cfg.is_posthog_configured = True

        with pytest.raises(RuntimeError, match="boom"):
            await runtime._run_node_async(
                node,
                state={},
                config={},
                locals_ctx=None,
                context=ctx_runtime,
            )

        mock_capture.assert_called_once()
        props = mock_capture.call_args.kwargs["properties"]
        assert props["success"] is False
        assert "boom" in props["error"]


@pytest.mark.asyncio
async def test_run_node_skips_posthog_when_no_user_context():
    """No PostHog event fires when runtime_context has no user."""
    from seer.core.runtime.nodes import NodeRuntime

    services = MagicMock()
    services.type_env.as_dict.return_value = {}
    runtime = NodeRuntime(services)

    node = MagicMock()
    node.id = "n3"
    node.type = "tool"

    impl = MagicMock()
    impl.execute_async = AsyncMock(return_value={"n3": "ok"})
    impl.get_analytics_properties = MagicMock(return_value={})

    with (
        patch("seer.core.nodes.registry.node_type_registry") as mock_registry,
        patch("seer.analytics.workflow_tracking.capture_event") as mock_capture,
    ):
        mock_registry.get.return_value = impl

        # runtime_context with no user
        ctx_runtime = MagicMock()
        ctx_runtime.user = None

        await runtime._run_node_async(
            node,
            state={},
            config={},
            locals_ctx=None,
            context=ctx_runtime,
        )

        mock_capture.assert_not_called()
