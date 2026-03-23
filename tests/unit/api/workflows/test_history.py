"""
Unit tests for workflow history operations logic.

Tests the core logic of workflow history operations from
seer.api.workflows.services.history module.
"""
from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock

import pytest

from seer.core.schema.models import (
    AgentNode,
    Edge,
    EdgeType,
    InlineSchema,
    Node,
    OutputContract,
    OutputMode,
    ToolNode,
    WorkflowSpec,
)
from seer.database import WorkflowRunSource, WorkflowRunStatus
from seer.api.workflows.services.history import (
    _build_node_label,
    _build_node_trace_from_value,
    _find_node_in_spec,
    _enrich_with_tool_node,
    _enrich_node_with_spec,
    _collect_graph_nodes,
    _build_execution_graph,
    _serialize_datetime,
    _snapshot_to_dict,
    _parse_workflow_spec,
    _build_trigger_info,
    _build_history_response,
    _get_error_traces_from_database,
    _merge_checkpoint_and_database_traces,
)
from tests.unit.helpers import utcnow


# =============================================================================
# Node Label Building Tests
# =============================================================================


@pytest.mark.unit
class TestBuildNodeLabel:
    """Tests for _build_node_label function."""

    def test_build_node_label_tool_node(self):
        """Test building label for a ToolNode includes tool name."""
        node = ToolNode(
            id="my_tool",
            tool="github.create_issue",
            inputs={"title": "Test"}
        )
        label = _build_node_label(node)
        assert label == "my_tool (github.create_issue)"

    def test_build_node_label_agent_node(self):
        """Test building label for AgentNode returns node ID."""
        node = AgentNode(
            id="ai_agent",
            inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Test prompt"}
        )
        label = _build_node_label(node)
        assert label == "ai_agent"

    def test_build_node_label_tool_node_with_complex_tool_name(self):
        """Test label with complex tool identifier."""
        node = ToolNode(
            id="n1",
            tool="slack.post_message_v2",
            inputs={}
        )
        label = _build_node_label(node)
        assert label == "n1 (slack.post_message_v2)"


# =============================================================================
# Find Node In Spec Tests
# =============================================================================


@pytest.mark.unit
class TestFindNodeInSpec:
    """Tests for _find_node_in_spec function."""

    def test_find_node_in_spec_found(self):
        """Test finding existing node in spec nodes list."""
        nodes: List[Node] = [
            ToolNode(id="n1", tool="test.tool", inputs={}),
            AgentNode(id="n2", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"}),
        ]

        found = _find_node_in_spec(nodes, "n1")

        assert found is not None
        assert found.id == "n1"
        assert isinstance(found, ToolNode)

    def test_find_node_in_spec_not_found(self):
        """Test handling when node is not found in spec."""
        nodes: List[Node] = [
            ToolNode(id="n1", tool="test.tool", inputs={}),
        ]

        found = _find_node_in_spec(nodes, "n99")

        assert found is None

    def test_find_node_in_spec_empty_list(self):
        """Test with empty nodes list."""
        nodes: List[Node] = []

        found = _find_node_in_spec(nodes, "n1")

        assert found is None

    def test_find_node_in_spec_llm_node(self):
        """Test finding LLM node."""
        nodes: List[Node] = [
            ToolNode(id="tool_1", tool="test.tool", inputs={}),
            AgentNode(id="llm_1", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"}),
        ]

        found = _find_node_in_spec(nodes, "llm_1")

        assert found is not None
        assert found.id == "llm_1"
        assert isinstance(found, AgentNode)


# =============================================================================
# Node Enrichment Tests
# =============================================================================


@pytest.mark.unit
class TestEnrichWithToolNode:
    """Tests for _enrich_with_tool_node function."""

    def test_enrich_with_tool_node_basic(self):
        """Test enriching dict with ToolNode metadata."""
        # Create a real ToolNode instance
        node = ToolNode(
            id="tool-1",
            tool="github.create_issue",
            inputs={},
            expect_outputs=None
        )

        enriched = {}
        _enrich_with_tool_node(enriched, node)

        assert enriched["tool_name"] == "github.create_issue"
        assert "expect_outputs" not in enriched

    def test_enrich_with_tool_node_with_expect_output(self):
        """Test enriching with expect_outputs present."""
        node = ToolNode(
            id="tool-1",
            tool="test.tool",
            inputs={},
            expect_outputs=OutputContract(
                mode=OutputMode.json,
                schema=InlineSchema(schema={"type": "object", "properties": {"result": {"type": "string"}}})
            )
        )

        enriched = {}
        _enrich_with_tool_node(enriched, node)

        assert enriched["tool_name"] == "test.tool"
        assert "expect_outputs" in enriched
        assert enriched["expect_outputs"]["mode"] == "json"

    def test_enrich_with_tool_node_none_expect_outputs(self):
        """Test enriching when expect_outputs is None."""
        node = ToolNode(
            id="tool-1",
            tool="test.tool",
            inputs={}
        )

        enriched = {}
        _enrich_with_tool_node(enriched, node)

        assert enriched["tool_name"] == "test.tool"
        assert "expect_outputs" not in enriched


@pytest.mark.unit
class TestEnrichNodeWithSpec:
    """Tests for _enrich_node_with_spec function."""

    def test_enrich_node_with_spec_no_spec(self):
        """Test enrichment returns copy when no spec provided."""
        node_trace = {"node_id": "n1", "status": "completed"}

        enriched = _enrich_node_with_spec(node_trace, "n1", None)

        assert enriched == node_trace
        assert enriched is not node_trace  # Should be a copy

    def test_enrich_node_with_spec_node_not_found(self):
        """Test enrichment when node not found in spec."""
        node_trace = {"node_id": "n99", "status": "completed"}
        spec = WorkflowSpec(
            version="2",
            nodes=[ToolNode(id="n1", tool="test.tool", inputs={})],
            edges=[],
        )

        enriched = _enrich_node_with_spec(node_trace, "n99", spec)

        assert enriched == node_trace

    def test_enrich_node_with_spec_creates_copy(self):
        """Test that enrichment creates a copy without modifying original."""
        original_trace = {"node_id": "n99", "status": "completed", "output": "result"}

        # Use None spec to avoid enrichment logic that depends on missing attributes
        enriched = _enrich_node_with_spec(original_trace, "n99", None)

        # Should be a different object
        assert enriched is not original_trace
        # But with same content
        assert enriched == original_trace

    def test_enrich_node_with_spec_returns_copy_when_node_missing(self):
        """Test enrichment returns unmodified copy when node not in spec."""
        original_trace = {"node_id": "missing", "status": "completed", "data": {"key": "value"}}
        spec = WorkflowSpec(
            version="2",
            nodes=[ToolNode(id="other", tool="test.tool", inputs={})],
            edges=[],
        )

        enriched = _enrich_node_with_spec(original_trace, "missing", spec)

        # Original should be unchanged
        assert original_trace == {"node_id": "missing", "status": "completed", "data": {"key": "value"}}
        # Enriched should be a copy with same data
        assert enriched == original_trace
        assert enriched is not original_trace


# =============================================================================
# Graph Collection Tests
# =============================================================================


@pytest.mark.unit
class TestCollectGraphNodes:
    """Tests for _collect_graph_nodes function."""

    def test_collect_graph_nodes_tool_node(self):
        """Test collecting tool node info for graph."""
        node = ToolNode(id="my_tool", tool="github.issues", inputs={})
        nodes_list = []

        _collect_graph_nodes(node, nodes_list)

        assert len(nodes_list) == 1
        assert nodes_list[0]["id"] == "my_tool"
        assert nodes_list[0]["type"] == "tool"
        assert nodes_list[0]["label"] == "my_tool (github.issues)"

    def test_collect_graph_nodes_llm_node(self):
        """Test collecting LLM node info for graph."""
        node = AgentNode(id="ai_chat", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"})
        nodes_list = []

        _collect_graph_nodes(node, nodes_list)

        assert len(nodes_list) == 1
        assert nodes_list[0]["id"] == "ai_chat"
        assert nodes_list[0]["type"] == "agent"
        assert nodes_list[0]["label"] == "ai_chat"

    def test_collect_graph_nodes_multiple(self):
        """Test collecting multiple nodes."""
        nodes = [
            ToolNode(id="t1", tool="tool1", inputs={}),
            AgentNode(id="l1", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"}),
        ]
        nodes_list = []

        for node in nodes:
            _collect_graph_nodes(node, nodes_list)

        assert len(nodes_list) == 2
        assert nodes_list[0]["id"] == "t1"
        assert nodes_list[1]["id"] == "l1"


# =============================================================================
# Execution Graph Building Tests
# =============================================================================


@pytest.mark.unit
class TestBuildExecutionGraph:
    """Tests for _build_execution_graph function."""

    def test_build_execution_graph_none_spec(self):
        """Test building graph with None spec returns empty graph."""
        graph = _build_execution_graph(None)

        assert graph == {"nodes": [], "edges": []}

    def test_build_execution_graph_empty_spec(self):
        """Test building graph with empty nodes/edges."""
        spec = WorkflowSpec(version="2", nodes=[], edges=[])

        graph = _build_execution_graph(spec)

        assert graph == {"nodes": [], "edges": []}

    def test_build_execution_graph_with_nodes_and_edges(self):
        """Test building complete execution graph."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ToolNode(id="n1", tool="tool1", inputs={}),
                AgentNode(id="n2", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"}),
            ],
            edges=[
                Edge(source="n1", target="n2", type=EdgeType.default),
            ],
        )

        graph = _build_execution_graph(spec)

        assert len(graph["nodes"]) == 2
        assert graph["nodes"][0]["id"] == "n1"
        assert graph["nodes"][0]["type"] == "tool"
        assert graph["nodes"][1]["id"] == "n2"
        assert graph["nodes"][1]["type"] == "agent"

        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["source"] == "n1"
        assert graph["edges"][0]["target"] == "n2"

    def test_build_execution_graph_multiple_edges(self):
        """Test graph with multiple edges."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ToolNode(id="start", tool="tool1", inputs={}),
                ToolNode(id="branch_a", tool="tool2", inputs={}),
                ToolNode(id="branch_b", tool="tool3", inputs={}),
            ],
            edges=[
                Edge(source="start", target="branch_a", type=EdgeType.conditional_true),
                Edge(source="start", target="branch_b", type=EdgeType.conditional_false),
            ],
        )

        graph = _build_execution_graph(spec)

        assert len(graph["nodes"]) == 3
        assert len(graph["edges"]) == 2


# =============================================================================
# Datetime Serialization Tests
# =============================================================================


@pytest.mark.unit
class TestSerializeDatetime:
    """Tests for _serialize_datetime function."""

    def test_serialize_datetime_none(self):
        """Test serializing None returns None."""
        result = _serialize_datetime(None)
        assert result is None

    def test_serialize_datetime_valid(self):
        """Test serializing valid datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        result = _serialize_datetime(dt)

        assert result == "2024-01-15T10:30:00+00:00"

    def test_serialize_datetime_no_timezone(self):
        """Test serializing datetime without timezone."""
        dt = datetime(2024, 6, 20, 14, 45, 30)

        result = _serialize_datetime(dt)

        assert result == "2024-06-20T14:45:30"

    def test_serialize_datetime_invalid_type(self):
        """Test serializing invalid type returns None."""
        result = _serialize_datetime("not a datetime")
        assert result is None

    def test_serialize_datetime_object_without_isoformat(self):
        """Test serializing object without isoformat method."""
        result = _serialize_datetime(12345)
        assert result is None


# =============================================================================
# Snapshot to Dict Tests
# =============================================================================


@pytest.mark.unit
class TestSnapshotToDict:
    """Tests for _snapshot_to_dict function."""

    def test_snapshot_to_dict_basic(self):
        """Test converting snapshot with basic attributes."""
        snapshot = MagicMock()
        snapshot.checkpoint_id = "cp_123"
        snapshot.values = {"key": "value"}
        snapshot.metadata = {"run_id": "run_1"}
        # Remove attributes that shouldn't be present
        del snapshot.parent_checkpoint_id
        del snapshot.next
        del snapshot.tasks
        del snapshot.created_at
        del snapshot.config
        del snapshot.parent_config

        result = _snapshot_to_dict(snapshot)

        assert result["checkpoint_id"] == "cp_123"
        assert result["values"] == {"key": "value"}
        assert result["metadata"] == {"run_id": "run_1"}

    def test_snapshot_to_dict_all_fields(self):
        """Test converting snapshot with all supported fields."""
        snapshot = MagicMock()
        snapshot.checkpoint_id = "cp_456"
        snapshot.parent_checkpoint_id = "cp_455"
        snapshot.values = {"state": "data"}
        snapshot.next = ["node_2"]
        snapshot.tasks = []
        snapshot.metadata = {}
        snapshot.created_at = "2024-01-01T00:00:00Z"
        snapshot.config = {"thread_id": "t1"}
        snapshot.parent_config = {"thread_id": "t0"}

        result = _snapshot_to_dict(snapshot)

        assert result["checkpoint_id"] == "cp_456"
        assert result["parent_checkpoint_id"] == "cp_455"
        assert result["values"] == {"state": "data"}
        assert result["next"] == ["node_2"]
        assert result["tasks"] == []
        assert result["metadata"] == {}
        assert result["created_at"] == "2024-01-01T00:00:00Z"
        assert result["config"] == {"thread_id": "t1"}
        assert result["parent_config"] == {"thread_id": "t0"}

    def test_snapshot_to_dict_excludes_none_values(self):
        """Test that None values are excluded from result."""
        snapshot = MagicMock()
        snapshot.checkpoint_id = "cp_789"
        snapshot.values = None
        snapshot.metadata = {"key": "val"}
        del snapshot.parent_checkpoint_id
        del snapshot.next
        del snapshot.tasks
        del snapshot.created_at
        del snapshot.config
        del snapshot.parent_config

        result = _snapshot_to_dict(snapshot)

        assert result["checkpoint_id"] == "cp_789"
        assert "values" not in result  # None excluded
        assert result["metadata"] == {"key": "val"}

    def test_snapshot_to_dict_empty_snapshot(self):
        """Test converting snapshot with no matching attributes."""
        snapshot = MagicMock(spec=[])  # No attributes

        result = _snapshot_to_dict(snapshot)

        assert result == {}


# =============================================================================
# Workflow Spec Parsing Tests
# =============================================================================


@pytest.mark.unit
class TestParseWorkflowSpec:
    """Tests for _parse_workflow_spec function."""

    def test_parse_workflow_spec_valid(self):
        """Test parsing valid workflow spec from run."""
        run = MagicMock()
        run.spec = {
            "version": "2",
            "nodes": [{"id": "n1", "type": "tool", "tool": "test.tool", "inputs": {}}],
            "edges": [],
        }

        spec = _parse_workflow_spec(run)

        assert spec is not None
        assert spec.version == "2"
        assert len(spec.nodes) == 1
        assert spec.nodes[0].id == "n1"

    def test_parse_workflow_spec_invalid(self):
        """Test parsing invalid spec returns None."""
        run = MagicMock()
        run.spec = "not a dict"

        spec = _parse_workflow_spec(run)

        assert spec is None

    def test_parse_workflow_spec_defaults_empty_nodes(self):
        """Test parsing spec defaults to empty nodes/edges when not provided."""
        run = MagicMock()
        run.spec = {"version": "2"}  # nodes/edges default to empty lists

        spec = _parse_workflow_spec(run)

        # WorkflowSpec allows empty nodes/edges by default
        assert spec is not None
        assert spec.nodes == []
        assert spec.edges == []

    def test_parse_workflow_spec_invalid_node_type(self):
        """Test parsing spec with invalid node type returns None."""
        run = MagicMock()
        run.spec = {
            "version": "2",
            "nodes": [{"id": "n1", "type": "invalid_type"}],  # Invalid node type
            "edges": [],
        }

        spec = _parse_workflow_spec(run)

        assert spec is None

    def test_parse_workflow_spec_complex(self):
        """Test parsing complex workflow spec."""
        run = MagicMock()
        run.spec = {
            "version": "2",
            "nodes": [
                {"id": "tool_1", "type": "tool", "tool": "github.issues", "inputs": {"repo": "test"}},
                {"id": "llm_1", "type": "agent", "inputs": {"model": "qwen/qwen3-235b-a22b-2507", "prompt": "Analyze"}},
            ],
            "edges": [
                {"source": "tool_1", "target": "llm_1", "type": "default"}
            ],
        }

        spec = _parse_workflow_spec(run)

        assert spec is not None
        assert len(spec.nodes) == 2
        assert len(spec.edges) == 1
        assert isinstance(spec.nodes[0], ToolNode)
        assert isinstance(spec.nodes[1], AgentNode)


# =============================================================================
# Trigger Info Building Tests
# =============================================================================


@pytest.mark.unit
class TestBuildTriggerInfo:
    """Tests for _build_trigger_info function."""

    def _make_run(self, source=WorkflowRunSource.MANUAL, subscription=None, trigger_event=None):
        run = MagicMock()
        run.source = source
        run.subscription = subscription
        run.trigger_event = trigger_event
        return run

    def test_manual_run_returns_source_only(self):
        """Manual run returns only source=manual."""
        run = self._make_run(source=WorkflowRunSource.MANUAL)
        result = _build_trigger_info(run)
        assert result == {"source": "manual"}

    def test_manual_run_string_source(self):
        """Plain string 'manual' source is handled correctly."""
        run = self._make_run(source="manual")
        result = _build_trigger_info(run)
        assert result == {"source": "manual"}

    def test_trigger_run_no_subscription_no_event(self):
        """Trigger source with no subscription or event returns only source=trigger."""
        run = self._make_run(source=WorkflowRunSource.TRIGGER, subscription=None, trigger_event=None)
        result = _build_trigger_info(run)
        assert result == {"source": "trigger"}

    def test_trigger_run_with_subscription_only(self):
        """Trigger run with subscription but no event returns subscription fields."""
        subscription = MagicMock()
        subscription.trigger_id = "abc123"
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.title = "Gmail Inbox"
        run = self._make_run(source=WorkflowRunSource.TRIGGER, subscription=subscription, trigger_event=None)

        result = _build_trigger_info(run)

        assert result["source"] == "trigger"
        assert result["trigger_id"] == "abc123"
        assert result["trigger_key"] == "poll.gmail.email_received"
        assert result["title"] == "Gmail Inbox"
        assert "event_data" not in result
        assert "occurred_at" not in result

    def test_trigger_run_with_full_data(self):
        """Trigger run with both subscription and event returns full response."""
        from datetime import datetime, timezone
        subscription = MagicMock()
        subscription.trigger_id = "abc123"
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.title = "Gmail Inbox"

        trigger_event = MagicMock()
        trigger_event.trigger_key = "poll.gmail.email_received"
        trigger_event.event = {"data": {"subject": "Hello", "from": "test@example.com"}}
        trigger_event.occurred_at = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        trigger_event.received_at = datetime(2024, 6, 1, 12, 0, 1, tzinfo=timezone.utc)

        run = self._make_run(source=WorkflowRunSource.TRIGGER, subscription=subscription, trigger_event=trigger_event)
        result = _build_trigger_info(run)

        assert result["source"] == "trigger"
        assert result["trigger_id"] == "abc123"
        assert result["trigger_key"] == "poll.gmail.email_received"
        assert result["title"] == "Gmail Inbox"
        assert result["event_data"] == {"subject": "Hello", "from": "test@example.com"}
        assert result["occurred_at"] == "2024-06-01T12:00:00+00:00"
        assert result["received_at"] == "2024-06-01T12:00:01+00:00"

    def test_trigger_run_event_with_null_occurred_at(self):
        """Trigger event with occurred_at=None serializes to None."""
        trigger_event = MagicMock()
        trigger_event.trigger_key = "poll.gmail.email_received"
        trigger_event.event = {"data": {}}
        trigger_event.occurred_at = None
        trigger_event.received_at = None

        run = self._make_run(source=WorkflowRunSource.TRIGGER, trigger_event=trigger_event, subscription=None)
        result = _build_trigger_info(run)

        assert result["occurred_at"] is None
        assert result["received_at"] is None

    def test_trigger_run_event_with_empty_event_envelope(self):
        """Event envelope with no 'data' key returns event_data=None."""
        trigger_event = MagicMock()
        trigger_event.trigger_key = "poll.gmail.email_received"
        trigger_event.event = {}
        trigger_event.occurred_at = None
        trigger_event.received_at = None

        run = self._make_run(source=WorkflowRunSource.TRIGGER, trigger_event=trigger_event, subscription=None)
        result = _build_trigger_info(run)

        assert result["event_data"] is None

    def test_trigger_run_fallback_trigger_key_from_event(self):
        """When no subscription, trigger_key is taken from the event."""
        trigger_event = MagicMock()
        trigger_event.trigger_key = "webhook.github"
        trigger_event.event = {"data": {"action": "push"}}
        trigger_event.occurred_at = None
        trigger_event.received_at = None

        run = self._make_run(source=WorkflowRunSource.TRIGGER, subscription=None, trigger_event=trigger_event)
        result = _build_trigger_info(run)

        assert result["trigger_key"] == "webhook.github"
        assert "trigger_id" not in result

    def test_manual_run_with_trigger_event_returns_event_data(self):
        """Manual run WITH a linked TriggerEvent surfaces trigger data (trigger_event_override flow)."""
        from datetime import datetime, timezone

        trigger_event = MagicMock()
        trigger_event.trigger_key = "poll.gmail.email_received"
        trigger_event.event = {"trigger_key": "poll.gmail.email_received", "data": {"subject": "Hello", "from": "test@example.com"}}
        trigger_event.occurred_at = None
        trigger_event.received_at = datetime(2026, 3, 3, 7, 39, 19, tzinfo=timezone.utc)

        run = self._make_run(source=WorkflowRunSource.MANUAL, subscription=None, trigger_event=trigger_event)
        result = _build_trigger_info(run)

        assert result["source"] == "manual"
        assert result["trigger_key"] == "poll.gmail.email_received"
        assert result["event_data"] == {"subject": "Hello", "from": "test@example.com"}
        assert result["occurred_at"] is None
        assert result["received_at"] == "2026-03-03T07:39:19+00:00"
        assert "trigger_id" not in result

    def test_manual_run_with_trigger_event_and_subscription(self):
        """Manual run with linked subscription AND trigger_event returns full metadata."""
        subscription = MagicMock()
        subscription.trigger_id = "abc123"
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.title = "Gmail Inbox"

        trigger_event = MagicMock()
        trigger_event.trigger_key = "poll.gmail.email_received"
        trigger_event.event = {"data": {"subject": "Test"}}
        trigger_event.occurred_at = None
        trigger_event.received_at = None

        run = self._make_run(source=WorkflowRunSource.MANUAL, subscription=subscription, trigger_event=trigger_event)
        result = _build_trigger_info(run)

        assert result["source"] == "manual"
        assert result["trigger_id"] == "abc123"
        assert result["trigger_key"] == "poll.gmail.email_received"
        assert result["title"] == "Gmail Inbox"
        assert result["event_data"] == {"subject": "Test"}


# =============================================================================
# History Response Building Tests
# =============================================================================


@pytest.mark.unit
class TestBuildHistoryResponse:
    """Tests for _build_history_response function."""

    def test_build_history_response_basic(self):
        """Test building basic history response."""
        run = MagicMock()
        run.run_id = "run_123"
        run.workflow = MagicMock()
        run.workflow.workflow_id = "wf_456"
        run.status = WorkflowRunStatus.SUCCEEDED
        run.created_at = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        run.started_at = datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
        run.finished_at = datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)

        nodes = [{"node_id": "n1", "status": "completed"}]

        response = _build_history_response(run, nodes, None)

        assert len(response) == 1
        assert response[0]["run_id"] == "run_123"
        assert response[0]["workflow_id"] == "wf_456"
        assert response[0]["status"] == "succeeded"
        assert response[0]["created_at"] == "2024-01-01T00:00:00+00:00"
        assert response[0]["started_at"] == "2024-01-01T00:00:01+00:00"
        assert response[0]["finished_at"] == "2024-01-01T00:01:00+00:00"
        assert response[0]["nodes"] == nodes
        assert response[0]["execution_graph"] == {"nodes": [], "edges": []}

    def test_build_history_response_no_workflow(self):
        """Test building response when run has no workflow."""
        run = MagicMock()
        run.run_id = "run_orphan"
        run.workflow = None
        run.status = "running"
        run.created_at = None
        run.started_at = None
        run.finished_at = None

        response = _build_history_response(run, [], None)

        assert response[0]["run_id"] == "run_orphan"
        assert response[0]["workflow_id"] is None
        assert response[0]["status"] == "running"
        assert response[0]["nodes"] == []

    def test_build_history_response_with_spec(self):
        """Test building response with workflow spec for execution graph."""
        run = MagicMock()
        run.run_id = "run_with_graph"
        run.workflow = MagicMock()
        run.workflow.workflow_id = "wf_1"
        run.status = MagicMock()
        run.status.value = "completed"
        run.created_at = None
        run.started_at = None
        run.finished_at = None

        spec = WorkflowSpec(
            version="2",
            nodes=[
                ToolNode(id="n1", tool="test.tool", inputs={}),
                AgentNode(id="n2", inputs={"model": "qwen/qwen3-235b-a22b-2507", "prompt": "test"}),
            ],
            edges=[Edge(source="n1", target="n2")],
        )

        nodes = [
            {"node_id": "n1", "status": "completed"},
            {"node_id": "n2", "status": "completed"},
        ]

        response = _build_history_response(run, nodes, spec)

        assert len(response[0]["execution_graph"]["nodes"]) == 2
        assert len(response[0]["execution_graph"]["edges"]) == 1
        assert response[0]["nodes"] == nodes

    def test_build_history_response_status_enum(self):
        """Test response handles WorkflowRunStatus enum correctly."""
        run = MagicMock()
        run.run_id = "run_enum"
        run.workflow = None
        run.status = WorkflowRunStatus.FAILED
        run.created_at = None
        run.started_at = None
        run.finished_at = None

        response = _build_history_response(run, [], None)

        assert response[0]["status"] == "failed"

    def test_build_history_response_status_string(self):
        """Test response handles string status."""
        run = MagicMock()
        run.run_id = "run_str"
        run.workflow = None
        run.status = "cancelled"
        run.created_at = None
        run.started_at = None
        run.finished_at = None
        run.source = WorkflowRunSource.MANUAL
        run.subscription = None
        run.trigger_event = None

        response = _build_history_response(run, [], None)

        assert response[0]["status"] == "cancelled"

    def test_build_history_response_includes_trigger_key_for_manual(self):
        """Verify 'trigger' key is present and manual for non-trigger runs."""
        run = MagicMock()
        run.run_id = "run_manual"
        run.workflow = None
        run.status = WorkflowRunStatus.SUCCEEDED
        run.created_at = None
        run.started_at = None
        run.finished_at = None
        run.source = WorkflowRunSource.MANUAL
        run.subscription = None
        run.trigger_event = None

        response = _build_history_response(run, [], None)

        assert "trigger" in response[0]
        assert response[0]["trigger"] == {"source": "manual"}

    def test_build_history_response_includes_trigger_key_for_trigger_run(self):
        """Verify 'trigger' key is populated for trigger-initiated runs."""
        subscription = MagicMock()
        subscription.trigger_id = "sub_xyz"
        subscription.trigger_key = "poll.gmail.email_received"
        subscription.title = "My Gmail"

        run = MagicMock()
        run.run_id = "run_triggered"
        run.workflow = None
        run.status = WorkflowRunStatus.SUCCEEDED
        run.created_at = None
        run.started_at = None
        run.finished_at = None
        run.source = WorkflowRunSource.TRIGGER
        run.subscription = subscription
        run.trigger_event = None

        response = _build_history_response(run, [], None)

        assert "trigger" in response[0]
        assert response[0]["trigger"]["source"] == "trigger"
        assert response[0]["trigger"]["trigger_id"] == "sub_xyz"
        assert response[0]["trigger"]["trigger_key"] == "poll.gmail.email_received"
        assert response[0]["trigger"]["title"] == "My Gmail"


# =============================================================================
# Run Status Tests
# =============================================================================


@pytest.mark.unit
class TestRunStatusValues:
    """Tests for run status handling."""

    def test_valid_run_status_values(self):
        """Test that WorkflowRunStatus has expected values."""
        # Verify expected enum members exist with correct values
        expected = {
            "QUEUED": "queued",
            "RUNNING": "running",
            "SUCCEEDED": "succeeded",
            "FAILED": "failed",
            "CANCELLED": "cancelled",
        }

        for enum_name, expected_value in expected.items():
            status = getattr(WorkflowRunStatus, enum_name, None)
            assert status is not None, f"Missing status: {enum_name}"
            assert status.value == expected_value


# =============================================================================
# Integration-style Unit Tests
# =============================================================================


@pytest.mark.unit
class TestHistoryWorkflowIntegration:
    """Integration-style tests combining multiple history functions."""

    def test_full_enrichment_flow_with_node_not_found(self):
        """Test enrichment flow when nodes not found in spec returns copies."""
        # Create a workflow spec with different node IDs
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ToolNode(id="other_node", tool="api.fetch", inputs={}),
            ],
            edges=[],
        )

        # Create node traces with IDs that don't match the spec
        traces = [
            {"node_id": "fetch_data", "status": "completed", "output": {"data": [1, 2, 3]}},
            {"node_id": "analyze", "status": "completed", "output": "Analysis complete"},
        ]

        # Enrich each trace - since nodes aren't found, should return copies unchanged
        enriched_traces = []
        for trace in traces:
            enriched = _enrich_node_with_spec(trace, trace["node_id"], spec)
            enriched_traces.append(enriched)

        # Verify enrichment preserved original data (no changes since nodes not found)
        assert enriched_traces[0]["node_id"] == "fetch_data"
        assert enriched_traces[0]["status"] == "completed"
        assert enriched_traces[0]["output"] == {"data": [1, 2, 3]}

        assert enriched_traces[1]["node_id"] == "analyze"
        assert enriched_traces[1]["status"] == "completed"

    def test_enrichment_with_mock_nodes(self):
        """Test enrichment flow using mock nodes with expected attributes."""
        # Create mock ToolNode with the attributes the enrichment function expects
        mock_tool_node = MagicMock()
        mock_tool_node.id = "fetch_data"
        mock_tool_node.tool = "api.fetch"
        mock_tool_node.expect_outputs = None

        # Test tool node enrichment
        tool_enriched = {}
        _enrich_with_tool_node(tool_enriched, mock_tool_node)
        assert tool_enriched["tool_name"] == "api.fetch"

    def test_execution_graph_matches_spec_structure(self):
        """Test that execution graph accurately represents spec structure."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ToolNode(id="step1", tool="tool.a", inputs={}),
                ToolNode(id="step2", tool="tool.b", inputs={}),
                ToolNode(id="step3", tool="tool.c", inputs={}),
            ],
            edges=[
                Edge(source="step1", target="step2"),
                Edge(source="step2", target="step3"),
            ],
        )

        graph = _build_execution_graph(spec)

        # Verify node count matches
        assert len(graph["nodes"]) == len(spec.nodes)

        # Verify edge count matches
        assert len(graph["edges"]) == len(spec.edges)

        # Verify node IDs are preserved
        node_ids = {n["id"] for n in graph["nodes"]}
        spec_node_ids = {n.id for n in spec.nodes}
        assert node_ids == spec_node_ids

        # Verify edge connections are preserved
        for i, edge in enumerate(graph["edges"]):
            assert edge["source"] == spec.edges[i].source
            assert edge["target"] == spec.edges[i].target

    def test_history_response_complete_structure(self):
        """Test that history response has complete expected structure."""
        run = MagicMock()
        run.run_id = "run_complete"
        run.workflow = MagicMock()
        run.workflow.workflow_id = "wf_complete"
        run.status = MagicMock()
        run.status.value = "completed"
        run.created_at = utcnow()
        run.started_at = utcnow()
        run.finished_at = utcnow()
        run.source = WorkflowRunSource.MANUAL
        run.subscription = None
        run.trigger_event = None

        spec = WorkflowSpec(
            version="2",
            nodes=[ToolNode(id="n1", tool="test", inputs={})],
            edges=[],
        )

        nodes = [{"node_id": "n1", "status": "completed", "output": "done"}]

        response = _build_history_response(run, nodes, spec)

        # Verify all expected keys are present
        expected_keys = {
            "run_id", "workflow_id", "status",
            "created_at", "started_at", "finished_at",
            "error", "nodes", "execution_graph", "trigger"
        }
        assert set(response[0].keys()) == expected_keys

        # Verify execution_graph structure
        assert "nodes" in response[0]["execution_graph"]
        assert "edges" in response[0]["execution_graph"]

        # Verify trigger structure for manual run
        assert response[0]["trigger"] == {"source": "manual"}


# =============================================================================
# Database Error Trace Extraction Tests
# =============================================================================


@pytest.mark.unit
class TestGetErrorTracesFromDatabase:
    """Tests for _get_error_traces_from_database function."""

    def test_get_error_traces_no_node_traces(self):
        """Test handling when node_traces is None."""
        run = MagicMock()
        run.node_traces = None

        traces = _get_error_traces_from_database(run)

        assert traces == []

    def test_get_error_traces_empty_dict(self):
        """Test handling when node_traces is empty dict."""
        run = MagicMock()
        run.node_traces = {}

        traces = _get_error_traces_from_database(run)

        assert traces == []

    def test_get_error_traces_single_trace_with_node_id(self):
        """Test extracting single trace when node_traces has node_id key."""
        run = MagicMock()
        run.node_traces = {
            "node_id": "kb_query-1",
            "node_type": "tool",
            "status": "failed",
            "error": {"type": "KeyError", "message": "'kb_id'"},
            "inputs": {"query": "test"},
            "timestamp": "2024-01-01T00:00:00Z",
        }

        traces = _get_error_traces_from_database(run)

        assert len(traces) == 1
        assert traces[0]["node_id"] == "kb_query-1"
        assert traces[0]["status"] == "failed"

    def test_get_error_traces_collection_of_traces(self):
        """Test extracting traces from collection keyed by trace_key."""
        run = MagicMock()
        run.node_traces = {
            "_trace_kb_query-1": {
                "node_id": "kb_query-1",
                "node_type": "tool",
                "status": "failed",
                "error": {"type": "KeyError", "message": "'kb_id'"},
            },
            "_trace_agent-1": {
                "node_id": "agent-1",
                "node_type": "agent",
                "status": "succeeded",  # Should be excluded
                "output": "Hello",
            },
        }

        traces = _get_error_traces_from_database(run)

        # Only failed traces should be included
        assert len(traces) == 1
        assert traces[0]["node_id"] == "kb_query-1"
        assert traces[0]["status"] == "failed"

    def test_get_error_traces_multiple_failed(self):
        """Test extracting multiple failed traces."""
        run = MagicMock()
        run.node_traces = {
            "_trace_node1": {
                "node_id": "node1",
                "status": "failed",
                "error": {"type": "Error1"},
            },
            "_trace_node2": {
                "node_id": "node2",
                "status": "failed",
                "error": {"type": "Error2"},
            },
        }

        traces = _get_error_traces_from_database(run)

        assert len(traces) == 2
        node_ids = {t["node_id"] for t in traces}
        assert node_ids == {"node1", "node2"}

    def test_get_error_traces_ignores_non_dict_values(self):
        """Test that non-dict values in node_traces are ignored."""
        run = MagicMock()
        run.node_traces = {
            "_trace_valid": {
                "node_id": "valid",
                "status": "failed",
            },
            "_trace_invalid": "not a dict",  # Should be ignored
        }

        traces = _get_error_traces_from_database(run)

        assert len(traces) == 1
        assert traces[0]["node_id"] == "valid"


# =============================================================================
# Trace Merging Tests
# =============================================================================


@pytest.mark.unit
class TestMergeCheckpointAndDatabaseTraces:
    """Tests for _merge_checkpoint_and_database_traces function."""

    def test_merge_empty_lists(self):
        """Test merging two empty lists."""
        result = _merge_checkpoint_and_database_traces([], [])
        assert result == []

    def test_merge_only_checkpoint_traces(self):
        """Test when only checkpoint traces exist."""
        checkpoint_traces = [
            {"node_id": "llm-1", "status": "succeeded", "output": "Hello"},
            {"node_id": "tool-1", "status": "succeeded", "output": {"data": 1}},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, [])

        assert len(result) == 2
        assert result == checkpoint_traces

    def test_merge_only_database_traces(self):
        """Test when only database traces exist."""
        db_traces = [
            {"node_id": "kb_query-1", "status": "failed", "error": {"type": "KeyError"}},
        ]

        result = _merge_checkpoint_and_database_traces([], db_traces)

        assert len(result) == 1
        assert result[0]["node_id"] == "kb_query-1"

    def test_merge_no_overlap(self):
        """Test merging when checkpoint and db traces have different nodes."""
        checkpoint_traces = [
            {"node_id": "llm-1", "status": "succeeded", "output": "Hello"},
        ]
        db_traces = [
            {"node_id": "kb_query-1", "status": "failed", "error": {"type": "KeyError"}},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        assert len(result) == 2
        node_ids = {t["node_id"] for t in result}
        assert node_ids == {"llm-1", "kb_query-1"}

    def test_merge_with_overlap_includes_failed_for_debugging(self):
        """Test that failed db traces are included even when checkpoint success exists.

        This behavior change is intentional: we now include failed traces for debugging
        purposes, even when a successful trace exists for the same node.
        """
        checkpoint_traces = [
            {"node_id": "node-1", "status": "succeeded", "output": "from_checkpoint"},
        ]
        db_traces = [
            {"node_id": "node-1", "status": "failed", "error": {"type": "Error"}},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        # Both traces should be present (success + failure for debugging)
        assert len(result) == 2
        # Checkpoint trace should come first
        assert result[0]["status"] == "succeeded"
        assert result[0]["output"] == "from_checkpoint"
        # Failed trace should be included for debugging
        assert result[1]["status"] == "failed"

    def test_merge_mixed_scenario(self):
        """Test realistic scenario with checkpoint success and db failure."""
        checkpoint_traces = [
            {"node_id": "llm-1", "status": "succeeded", "output": "Hi!"},
        ]
        db_traces = [
            {"node_id": "kb_query-1", "status": "failed", "error": {"type": "KeyError", "message": "'kb_id'"}},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        assert len(result) == 2

        # Find each trace
        llm_trace = next(t for t in result if t["node_id"] == "llm-1")
        kb_trace = next(t for t in result if t["node_id"] == "kb_query-1")

        assert llm_trace["status"] == "succeeded"
        assert kb_trace["status"] == "failed"
        assert kb_trace["error"]["type"] == "KeyError"

    def test_merge_preserves_order(self):
        """Test that checkpoint traces come before database traces."""
        checkpoint_traces = [
            {"node_id": "node-1", "status": "succeeded"},
            {"node_id": "node-2", "status": "succeeded"},
        ]
        db_traces = [
            {"node_id": "node-3", "status": "failed"},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        assert result[0]["node_id"] == "node-1"
        assert result[1]["node_id"] == "node-2"
        assert result[2]["node_id"] == "node-3"

    def test_merge_handles_missing_node_id(self):
        """Test handling of traces without node_id.

        Note: The current implementation includes failed traces regardless of
        whether they have a node_id. This is intentional to capture all errors
        for debugging.
        """
        checkpoint_traces = [
            {"node_id": "valid-1", "status": "succeeded"},
        ]
        db_traces = [
            {"status": "failed"},  # Missing node_id - still included since failed
            {"node_id": "valid-2", "status": "failed"},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        # All failed traces are included for debugging
        assert len(result) == 3
        assert result[0]["node_id"] == "valid-1"
        assert result[1]["status"] == "failed"  # Missing node_id but included
        assert result[2]["node_id"] == "valid-2"

    def test_merge_includes_failed_trace_when_success_exists(self):
        """Test that failed traces are included even when success exists for same node.

        This is critical for debugging loop iteration issues where the same node
        may succeed on some iterations but fail on others.
        """
        checkpoint_traces = [
            {"node_id": "log_sent_status", "status": "succeeded", "output": "E3", "timestamp": "2024-01-01T18:11:34Z"},
        ]
        db_traces = [
            {"node_id": "log_sent_status", "status": "failed", "error": {"type": "RemoteProtocolError"}, "timestamp": "2024-01-01T18:11:51Z"},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        # BOTH traces should be included for debugging
        assert len(result) == 2

        # Verify we have both success and failure
        statuses = {t["status"] for t in result}
        assert statuses == {"succeeded", "failed"}

        # Verify both are for the same node
        node_ids = {t["node_id"] for t in result}
        assert node_ids == {"log_sent_status"}

    def test_merge_includes_multiple_failed_traces_for_same_node(self):
        """Test that multiple failed traces for the same node are all included."""
        checkpoint_traces = [
            {"node_id": "api_call", "status": "succeeded", "output": "ok"},
        ]
        db_traces = [
            {"node_id": "api_call", "status": "failed", "error": {"type": "Error1"}, "timestamp": "T1"},
            {"node_id": "api_call", "status": "failed", "error": {"type": "Error2"}, "timestamp": "T2"},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        # All 3 traces should be included
        assert len(result) == 3

        # Verify 1 success and 2 failures
        succeeded = [t for t in result if t["status"] == "succeeded"]
        failed = [t for t in result if t["status"] == "failed"]
        assert len(succeeded) == 1
        assert len(failed) == 2

    def test_merge_loop_iteration_scenario(self):
        """Test realistic loop iteration scenario with mixed success/failure.

        Simulates a for_each loop where node succeeds on some iterations
        but fails on others.
        """
        # Checkpoint has successful traces from iteration 0 and 1
        checkpoint_traces = [
            {"node_id": "process", "status": "succeeded", "output": "done_0", "iteration": 0},
            {"node_id": "process", "status": "succeeded", "output": "done_1", "iteration": 1},
        ]
        # Database has failed trace from iteration 2
        db_traces = [
            {"node_id": "process", "status": "failed", "error": {"type": "NetworkError"}, "iteration": 2},
        ]

        result = _merge_checkpoint_and_database_traces(checkpoint_traces, db_traces)

        # All 3 traces should be included (2 successes + 1 failure)
        assert len(result) == 3

        iterations = [t.get("iteration") for t in result]
        assert set(iterations) == {0, 1, 2}


# =============================================================================
# _build_node_trace_from_value artifacts propagation tests
# =============================================================================


@pytest.mark.unit
class TestBuildNodeTraceFromValueArtifacts:
    """Tests that _build_node_trace_from_value propagates the artifacts field."""

    def _base_value(self, **kwargs):
        base = {
            "node_type": "agent",
            "inputs": {},
            "output": "Done",
            "timestamp": None,
            "output_key": "agent_1",
        }
        base.update(kwargs)
        return base

    def test_artifacts_included_when_present(self):
        """Artifacts list from checkpoint value should appear in the node trace."""
        artifacts = [
            {
                "_type": "workflow_file_ref",
                "file_id": "fid-1",
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 42000,
                "storage_path": "s3://bucket/run_1/report.pdf",
                "workflow_run_id": "run_1",
                "created_at": "2026-03-05T06:05:00+00:00",
            }
        ]
        value = self._base_value(artifacts=artifacts)
        trace = _build_node_trace_from_value(value, "agent_1")

        assert "artifacts" in trace
        assert len(trace["artifacts"]) == 1
        assert trace["artifacts"][0]["filename"] == "report.pdf"
        assert trace["artifacts"][0]["_type"] == "workflow_file_ref"

    def test_artifacts_excluded_when_absent(self):
        """Node trace should not have artifacts key when value has no artifacts."""
        value = self._base_value()
        trace = _build_node_trace_from_value(value, "agent_1")
        assert "artifacts" not in trace

    def test_artifacts_excluded_when_empty_list(self):
        """Node trace should not have artifacts key when value has empty artifacts list."""
        value = self._base_value(artifacts=[])
        trace = _build_node_trace_from_value(value, "agent_1")
        assert "artifacts" not in trace

    def test_multiple_artifacts_propagated(self):
        """All artifacts in the list should be propagated."""
        artifacts = [
            {"_type": "workflow_file_ref", "filename": "a.pdf"},
            {"_type": "workflow_file_ref", "filename": "b.docx"},
        ]
        value = self._base_value(artifacts=artifacts)
        trace = _build_node_trace_from_value(value, "agent_1")
        assert len(trace["artifacts"]) == 2
        filenames = {a["filename"] for a in trace["artifacts"]}
        assert filenames == {"a.pdf", "b.docx"}
