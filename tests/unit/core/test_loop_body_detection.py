"""
Unit tests for loop body node detection in lower_control_flow.py.

These tests verify that _find_loop_body_nodes correctly identifies all nodes
within a for_each loop, including nodes after nested control flow.
"""

from __future__ import annotations

import pytest

from seer.core.compiler.lower_control_flow import _find_loop_body_nodes, build_execution_plan
from seer.core.schema.models import (
    Edge,
    EdgeType,
    ForEachNode,
    IfNode,
    Node,
    ToolNode,
    WorkflowSpec,
)


class TestFindLoopBodyNodes:
    """Tests for _find_loop_body_nodes function."""

    def test_simple_linear_body(self):
        """Test detection of simple linear loop body: A -> B -> C."""
        # loop -> A -> B -> C (terminal)
        nodes = {
            "loop": ForEachNode(id="loop", items="${items}"),
            "node_a": ToolNode(id="node_a", tool="test.tool", inputs={}),
            "node_b": ToolNode(id="node_b", tool="test.tool", inputs={}),
            "node_c": ToolNode(id="node_c", tool="test.tool", inputs={}),
        }

        outgoing_edges = {
            "loop": [Edge(source="loop", target="node_a", type=EdgeType.loop_body)],
            "node_a": [Edge(source="node_a", target="node_b", type=EdgeType.default)],
            "node_b": [Edge(source="node_b", target="node_c", type=EdgeType.default)],
            "node_c": [],  # Terminal - no outgoing edges
        }

        body_nodes, terminal_nodes = _find_loop_body_nodes(
            loop_node_id="loop",
            body_entry_id="node_a",
            outgoing_edges=outgoing_edges,
            node_map=nodes,
        )

        assert body_nodes == {"node_a", "node_b", "node_c"}
        assert terminal_nodes == {"node_c"}

    def test_body_with_back_edge(self):
        """Test detection when body has explicit back-edge to loop."""
        # loop -> A -> B -> loop (back-edge)
        nodes = {
            "loop": ForEachNode(id="loop", items="${items}"),
            "node_a": ToolNode(id="node_a", tool="test.tool", inputs={}),
            "node_b": ToolNode(id="node_b", tool="test.tool", inputs={}),
        }

        outgoing_edges = {
            "loop": [Edge(source="loop", target="node_a", type=EdgeType.loop_body)],
            "node_a": [Edge(source="node_a", target="node_b", type=EdgeType.default)],
            "node_b": [Edge(source="node_b", target="loop", type=EdgeType.default)],  # Back-edge
        }

        body_nodes, terminal_nodes = _find_loop_body_nodes(
            loop_node_id="loop",
            body_entry_id="node_a",
            outgoing_edges=outgoing_edges,
            node_map=nodes,
        )

        assert body_nodes == {"node_a", "node_b"}
        assert terminal_nodes == set()  # No terminal nodes - all have back-edge

    def test_body_with_if_node(self):
        """Test detection includes if node and both branches.

        This tests the bug fix: nodes after if_node should be in body_nodes.
        """
        # loop -> A -> if_check -> (true) B -> C -> terminal
        #                       -> (false) D -> terminal
        nodes = {
            "loop": ForEachNode(id="loop", items="${items}"),
            "node_a": ToolNode(id="node_a", tool="test.tool", inputs={}),
            "if_check": IfNode(id="if_check", condition="${node_a} == 'yes'"),
            "node_b": ToolNode(id="node_b", tool="test.tool", inputs={}),
            "node_c": ToolNode(id="node_c", tool="test.tool", inputs={}),
            "node_d": ToolNode(id="node_d", tool="test.tool", inputs={}),
        }

        outgoing_edges = {
            "loop": [Edge(source="loop", target="node_a", type=EdgeType.loop_body)],
            "node_a": [Edge(source="node_a", target="if_check", type=EdgeType.default)],
            "if_check": [
                Edge(source="if_check", target="node_b", type=EdgeType.conditional_true),
                Edge(source="if_check", target="node_d", type=EdgeType.conditional_false),
            ],
            "node_b": [Edge(source="node_b", target="node_c", type=EdgeType.default)],
            "node_c": [],  # Terminal
            "node_d": [],  # Terminal
        }

        body_nodes, terminal_nodes = _find_loop_body_nodes(
            loop_node_id="loop",
            body_entry_id="node_a",
            outgoing_edges=outgoing_edges,
            node_map=nodes,
        )

        # All nodes should be in body_nodes (including if_check and nodes after it)
        assert body_nodes == {"node_a", "if_check", "node_b", "node_c", "node_d"}
        assert terminal_nodes == {"node_c", "node_d"}

    def test_body_with_nested_for_each(self):
        """Test detection includes nested for_each and nodes after it.

        This tests the bug fix: nodes after nested for_each should be in body_nodes.
        """
        # outer_loop -> A -> inner_loop -> B (inner body) -> inner_loop (back-edge)
        #                              -> C (after inner loop) -> terminal
        nodes = {
            "outer_loop": ForEachNode(id="outer_loop", items="${items}"),
            "node_a": ToolNode(id="node_a", tool="test.tool", inputs={}),
            "inner_loop": ForEachNode(id="inner_loop", items="${nested}"),
            "node_b": ToolNode(id="node_b", tool="test.tool", inputs={}),
            "node_c": ToolNode(id="node_c", tool="test.tool", inputs={}),
        }

        outgoing_edges = {
            "outer_loop": [Edge(source="outer_loop", target="node_a", type=EdgeType.loop_body)],
            "node_a": [Edge(source="node_a", target="inner_loop", type=EdgeType.default)],
            "inner_loop": [
                Edge(source="inner_loop", target="node_b", type=EdgeType.loop_body),
                Edge(source="inner_loop", target="node_c", type=EdgeType.loop_exit),
            ],
            "node_b": [Edge(source="node_b", target="inner_loop", type=EdgeType.default)],
            "node_c": [],  # Terminal
        }

        body_nodes, terminal_nodes = _find_loop_body_nodes(
            loop_node_id="outer_loop",
            body_entry_id="node_a",
            outgoing_edges=outgoing_edges,
            node_map=nodes,
        )

        # All nodes should be in body_nodes (including inner_loop and node_c after it)
        assert body_nodes == {"node_a", "inner_loop", "node_b", "node_c"}
        assert terminal_nodes == {"node_c"}

    def test_body_with_log_after_if(self):
        """Test the specific scenario from the RCA: log_sent_status after if node.

        This is the exact pattern that was causing the trace key collision bug.
        """
        # loop -> parse_lead -> completion_summary -> generate_email
        #      -> send_email -> log_sent_status -> terminal
        nodes = {
            "loop": ForEachNode(id="loop", items="${leads}"),
            "parse_lead": ToolNode(id="parse_lead", tool="llm.parse", inputs={}),
            "completion_summary": IfNode(id="completion_summary", condition="${status} == ''"),
            "generate_email": ToolNode(id="generate_email", tool="llm.generate", inputs={}),
            "send_email": ToolNode(id="send_email", tool="gmail.send", inputs={}),
            "log_sent_status": ToolNode(id="log_sent_status", tool="sheets.write", inputs={}),
        }

        outgoing_edges = {
            "loop": [Edge(source="loop", target="parse_lead", type=EdgeType.loop_body)],
            "parse_lead": [Edge(source="parse_lead", target="completion_summary", type=EdgeType.default)],
            "completion_summary": [
                Edge(source="completion_summary", target="generate_email", type=EdgeType.conditional_true),
            ],
            "generate_email": [Edge(source="generate_email", target="send_email", type=EdgeType.default)],
            "send_email": [Edge(source="send_email", target="log_sent_status", type=EdgeType.default)],
            "log_sent_status": [],  # Terminal
        }

        body_nodes, terminal_nodes = _find_loop_body_nodes(
            loop_node_id="loop",
            body_entry_id="parse_lead",
            outgoing_edges=outgoing_edges,
            node_map=nodes,
        )

        # CRITICAL: log_sent_status MUST be in body_nodes for proper trace key generation
        assert "log_sent_status" in body_nodes
        assert body_nodes == {
            "parse_lead",
            "completion_summary",
            "generate_email",
            "send_email",
            "log_sent_status",
        }
        assert terminal_nodes == {"log_sent_status"}


class TestBuildExecutionPlanLoopBodyMap:
    """Tests for loop_body_nodes in ExecutionPlan."""

    def test_execution_plan_includes_all_body_nodes(self):
        """Test that ExecutionPlan correctly populates loop_body_nodes."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ForEachNode(id="loop", items="${items}"),
                ToolNode(id="node_a", tool="test.tool", inputs={}),
                IfNode(id="if_check", condition="${node_a} == 'yes'"),
                ToolNode(id="node_b", tool="test.tool", inputs={}),
                ToolNode(id="node_c", tool="test.tool", inputs={}),
                ToolNode(id="exit", tool="test.tool", inputs={}),
            ],
            edges=[
                Edge(source="loop", target="node_a", type=EdgeType.loop_body),
                Edge(source="node_a", target="if_check", type=EdgeType.default),
                Edge(source="if_check", target="node_b", type=EdgeType.conditional_true),
                Edge(source="if_check", target="node_c", type=EdgeType.conditional_false),
                Edge(source="node_b", target="loop", type=EdgeType.default),  # Back-edge from true branch
                Edge(source="node_c", target="loop", type=EdgeType.default),  # Back-edge from false branch
                Edge(source="loop", target="exit", type=EdgeType.loop_exit),
            ],
        )

        plan = build_execution_plan(spec)

        # Verify loop_body_nodes is populated
        assert "loop" in plan.loop_body_nodes

        body_nodes = plan.loop_body_nodes["loop"]

        # All body nodes should be included
        assert "node_a" in body_nodes
        assert "if_check" in body_nodes
        assert "node_b" in body_nodes
        assert "node_c" in body_nodes

        # Exit node should NOT be in body_nodes
        assert "exit" not in body_nodes

    def test_execution_plan_terminal_nodes(self):
        """Test that ExecutionPlan correctly identifies terminal nodes."""
        spec = WorkflowSpec(
            version="2",
            nodes=[
                ForEachNode(id="loop", items="${items}"),
                ToolNode(id="process", tool="test.tool", inputs={}),
                ToolNode(id="log", tool="test.tool", inputs={}),
                ToolNode(id="exit", tool="test.tool", inputs={}),
            ],
            edges=[
                Edge(source="loop", target="process", type=EdgeType.loop_body),
                Edge(source="process", target="log", type=EdgeType.default),
                # log has no explicit back-edge - it's terminal
                Edge(source="loop", target="exit", type=EdgeType.loop_exit),
            ],
        )

        plan = build_execution_plan(spec)

        # Verify terminal nodes
        assert "loop" in plan.loop_terminal_nodes
        terminal_nodes = plan.loop_terminal_nodes["loop"]
        assert "log" in terminal_nodes
