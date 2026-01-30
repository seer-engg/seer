"""
Tests for Nexus agent primitive blocks documentation.
"""

from src.seer.agents.nexus.schema_context import generate_primitive_blocks_guide


def test_primitive_blocks_guide_generation():
    """Test that primitive blocks guide generates successfully."""
    guide = generate_primitive_blocks_guide()

    # Guide should be non-empty
    assert len(guide) > 1000, "Guide should contain substantial content"

    # Should contain all 5 primitive block types
    assert "TOOL BLOCK" in guide
    assert "LLM BLOCK" in guide
    assert "MCP BLOCK" in guide
    assert "IF BLOCK" in guide
    assert "FOR_EACH BLOCK" in guide


def test_primitive_blocks_guide_content():
    """Test that guide contains essential documentation elements."""
    guide = generate_primitive_blocks_guide()

    # Should have overview
    assert "Overview" in guide
    assert "5 primitive block types" in guide

    # Should document schemas
    assert "Schema:" in guide
    assert "Required Fields:" in guide

    # Should include examples
    assert "Example:" in guide
    assert '"type":' in guide  # JSON examples

    # Should document expression syntax
    assert "Expression Syntax Reference" in guide
    assert "${" in guide  # Expression syntax

    # Should have patterns section
    assert "Block Composition Patterns" in guide

    # Should have quick reference
    assert "Quick Reference Table" in guide


def test_primitive_blocks_guide_caching():
    """Test that guide function is properly cached."""
    guide1 = generate_primitive_blocks_guide()
    guide2 = generate_primitive_blocks_guide()

    # Should return same instance (cached)
    assert guide1 is guide2


def test_each_block_has_complete_documentation():
    """Test that each primitive block has comprehensive documentation."""
    guide = generate_primitive_blocks_guide()

    block_types = ["tool", "llm", "mcp", "if", "for_each"]

    for block_type in block_types:
        # Each block should have purpose
        assert f"**Purpose:**" in guide

        # Each block should have schema
        assert f'"type": "{block_type}"' in guide

        # Each block should have required fields section
        assert "**Required Fields:**" in guide

        # Each block should have examples
        assert "**Example:**" in guide or "**Examples:**" in guide


def test_guide_includes_edge_configurations():
    """Test that guide documents edge configurations for control flow blocks."""
    guide = generate_primitive_blocks_guide()

    # If block edge documentation
    assert "conditional_true" in guide
    assert "conditional_false" in guide

    # ForEach block edge documentation
    assert "loop_body" in guide
    assert "loop_exit" in guide


def test_guide_includes_output_modes():
    """Test that guide documents LLM output modes."""
    guide = generate_primitive_blocks_guide()

    # Should document both output modes
    assert '"mode": "text"' in guide
    assert '"mode": "json"' in guide

    # Should document JSON schema structure
    assert "json_schema" in guide


def test_guide_includes_common_use_cases():
    """Test that guide provides common use cases for each block."""
    guide = generate_primitive_blocks_guide()

    # Should have use cases section
    assert "**Common Use Cases:**" in guide or "Common Use Cases" in guide

    # Should mention practical scenarios
    assert "email" in guide.lower() or "gmail" in guide.lower()


def test_guide_expression_syntax_documentation():
    """Test that guide properly documents expression syntax."""
    guide = generate_primitive_blocks_guide()

    # Should document trigger data access
    assert "${trigger.data" in guide

    # Should document node output access
    assert "${node_id}" in guide

    # Should document loop variables
    assert "${item}" in guide
    assert "${index}" in guide

    # Should document operators
    assert "==" in guide
    assert "&&" in guide or "||" in guide
