"""
Tests for Nexus agent graph structure documentation.
"""

import pytest

from src.seer.agents.nexus.schema_context import generate_graph_structure_guide

pytestmark = pytest.mark.unit


def test_graph_structure_guide_generation():
    """Test that graph structure guide generates successfully."""
    guide = generate_graph_structure_guide()

    # Guide should be non-empty and substantial
    assert len(guide) > 5000, "Guide should contain substantial content"

    # Should contain key sections
    assert "Graph Structure" in guide
    assert "Compilation Pipeline" in guide


def test_graph_structure_guide_compilation_stages():
    """Test that guide documents all compilation stages."""
    guide = generate_graph_structure_guide()

    # Should document all 5 compilation stages
    assert "Parse" in guide
    assert "Build Type Environment" in guide
    assert "Validate References" in guide
    assert "Lower Control Flow" in guide
    assert "Emit LangGraph" in guide


def test_graph_structure_guide_fundamentals():
    """Test that guide covers fundamental graph concepts."""
    guide = generate_graph_structure_guide()

    # Should explain nodes, edges, and state
    assert "Nodes" in guide
    assert "Edges" in guide
    assert "State Management" in guide

    # Should explain state merging
    assert "merge" in guide.lower()
    assert "state_update" in guide or "state update" in guide.lower()


def test_graph_structure_guide_entry_exit_points():
    """Test that guide documents entry and exit points."""
    guide = generate_graph_structure_guide()

    # Should document START and END
    assert "START" in guide
    assert "END" in guide

    # Should document trigger routing
    assert "Trigger-Based Entry" in guide or "trigger routing" in guide.lower()
    assert "bootstrap" in guide.lower()


def test_graph_structure_guide_edge_types():
    """Test that guide documents all edge types."""
    guide = generate_graph_structure_guide()

    # Should document edge types
    assert "Default Edges" in guide or "default" in guide.lower()
    assert "Conditional Edges" in guide or "conditional" in guide.lower()
    assert "Loop Edges" in guide or "loop" in guide.lower()

    # Should mention specific edge type values
    assert "conditional_true" in guide
    assert "conditional_false" in guide
    assert "loop_body" in guide
    assert "loop_exit" in guide


def test_graph_structure_guide_multiple_edges():
    """Test that guide explains multiple edges behavior."""
    guide = generate_graph_structure_guide()

    # Should document multiple edges rules
    assert "Multiple Outgoing Edges" in guide or "multiple outgoing" in guide.lower()
    assert "Multiple Incoming Edges" in guide or "multiple incoming" in guide.lower()


def test_graph_structure_guide_diamond_pattern():
    """Test that guide explains diamond patterns."""
    guide = generate_graph_structure_guide()

    # Should document diamond pattern
    assert "Diamond" in guide

    # Should explain merge behavior
    assert "merge" in guide.lower()


def test_graph_structure_guide_loop_body_detection():
    """Test that guide explains loop body detection."""
    guide = generate_graph_structure_guide()

    # Should document loop body detection
    assert "Loop Body Detection" in guide or "loop body" in guide.lower()

    # Should explain implicit edges
    assert "implicit" in guide.lower()
    assert "terminal" in guide.lower()


def test_graph_structure_guide_constraints():
    """Test that guide documents key constraints."""
    guide = generate_graph_structure_guide()

    # Should have constraints section
    assert "Constraints" in guide or "Rules" in guide

    # Should mention unique IDs
    assert "unique" in guide.lower()

    # Should mention reachability
    assert "reachable" in guide.lower() or "orphaned" in guide.lower()


def test_graph_structure_guide_common_patterns():
    """Test that guide includes common graph patterns."""
    guide = generate_graph_structure_guide()

    # Should have patterns section
    assert "Common Patterns" in guide or "Pattern" in guide

    # Should include linear flow
    assert "Linear" in guide

    # Should include branching
    assert "Branch" in guide or "Conditional" in guide


def test_graph_structure_guide_debugging():
    """Test that guide includes debugging guidance."""
    guide = generate_graph_structure_guide()

    # Should have debugging section
    assert "Debugging" in guide or "Issue" in guide

    # Should mention common issues
    assert "not reachable" in guide.lower() or "orphaned" in guide.lower()


def test_graph_structure_guide_caching():
    """Test that guide function is properly cached."""
    guide1 = generate_graph_structure_guide()
    guide2 = generate_graph_structure_guide()

    # Should return same instance (cached)
    assert guide1 is guide2


def test_graph_structure_guide_compilation_details():
    """Test that guide explains compilation behavior."""
    guide = generate_graph_structure_guide()

    # Should explain LangGraph compilation
    assert "LangGraph" in guide
    assert "StateGraph" in guide

    # Should mention router functions
    assert "router" in guide.lower() or "routing" in guide.lower()


def test_graph_structure_guide_state_management():
    """Test that guide explains state management."""
    guide = generate_graph_structure_guide()

    # Should document state structure
    assert "state" in guide.lower()
    assert "node_id" in guide

    # Should explain trace keys
    assert "_trace" in guide or "trace" in guide.lower()


def test_graph_structure_guide_examples():
    """Test that guide includes code examples."""
    guide = generate_graph_structure_guide()

    # Should include JSON examples
    assert '"nodes":' in guide
    assert '"edges":' in guide

    # Should show example patterns
    assert "```" in guide or "```json" in guide


def test_graph_structure_guide_best_practices():
    """Test that guide includes best practices."""
    guide = generate_graph_structure_guide()

    # Should have best practices section
    assert "Best Practices" in guide

    # Should mention graph design principles
    assert "acyclic" in guide.lower() or "cycle" in guide.lower()
