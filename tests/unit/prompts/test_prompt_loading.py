"""
Unit tests for the prompt loading module.

Tests that prompts are correctly loaded from markdown files and
the loading utilities work as expected.
"""

import pytest
from pathlib import Path

from seer.prompts import (
    load_prompt,
    get_nexus_system_prompt,
    get_primitive_blocks_guide,
    get_graph_structure_guide,
    get_skill_guide,
    list_available_skills,
    clear_prompt_cache,
    PROMPTS_DIR,
)


@pytest.mark.unit
class TestPromptLoading:
    """Tests for the load_prompt function."""

    def test_load_prompt_success(self):
        """Test loading a valid prompt file."""
        # Load a known prompt
        content = load_prompt("nexus", "system_prompt")
        assert content is not None
        assert len(content) > 0
        assert "workflow assistant" in content.lower()

    def test_load_prompt_file_not_found(self):
        """Test that FileNotFoundError is raised for missing prompts."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", "missing_prompt")

    def test_load_prompt_caching(self):
        """Test that prompts are cached after first load."""
        # Clear cache first
        clear_prompt_cache()

        # First load
        content1 = load_prompt("nexus", "system_prompt")

        # Second load should return cached value
        content2 = load_prompt("nexus", "system_prompt")

        # Content should be identical
        assert content1 == content2

        # Verify cache is being used (check cache_info)
        cache_info = load_prompt.cache_info()
        assert cache_info.hits >= 1


@pytest.mark.unit
class TestNexusPrompts:
    """Tests for Nexus-specific prompt loaders."""

    def test_get_nexus_system_prompt(self):
        """Test loading the Nexus system prompt."""
        prompt = get_nexus_system_prompt()

        # Should contain key sections
        assert "Core Principles" in prompt
        assert "Tool Discovery" in prompt
        assert "WorkflowSpec" in prompt
        assert "Validation Checklist" in prompt

    def test_get_primitive_blocks_guide(self):
        """Test loading the primitive blocks guide."""
        guide = get_primitive_blocks_guide()

        # Should contain all block types
        assert "TOOL BLOCK" in guide
        assert "AGENT BLOCK" in guide
        assert "MCP BLOCK" in guide
        assert "IF BLOCK" in guide
        assert "FOR_EACH BLOCK" in guide

        # Should contain expression syntax
        assert "${" in guide
        assert "Expression Syntax" in guide

    def test_get_graph_structure_guide(self):
        """Test loading the graph structure guide."""
        guide = get_graph_structure_guide()

        # Should contain key sections
        assert "Compilation Pipeline" in guide
        assert "Entry Points" in guide
        assert "Edge Types" in guide
        assert "Diamond Pattern" in guide
        assert "Loop Body Detection" in guide


@pytest.mark.unit
class TestSkillGuides:
    """Tests for skill guide loading."""

    def test_get_gmail_skill_guide(self):
        """Test loading the Gmail skill guide."""
        guide = get_skill_guide("gmail")

        assert guide is not None
        assert "Gmail" in guide
        assert "gmail_create_draft" in guide or "send_email" in guide

    def test_get_skill_guide_not_found(self):
        """Test that None is returned for missing skill guides."""
        guide = get_skill_guide("nonexistent_integration")
        assert guide is None

    def test_get_skill_guide_case_insensitive(self):
        """Test that skill names are case-insensitive."""
        guide_lower = get_skill_guide("gmail")
        guide_upper = get_skill_guide("GMAIL")

        # Both should return the same content (or both None if not found)
        assert guide_lower == guide_upper

    def test_list_available_skills(self):
        """Test listing available skill guides."""
        skills = list_available_skills()

        # Should be a list
        assert isinstance(skills, list)

        # Gmail should be available (we copied it)
        assert "gmail" in skills


@pytest.mark.unit
class TestPromptDirectory:
    """Tests for prompt directory structure."""

    def test_prompts_dir_exists(self):
        """Test that the prompts directory exists."""
        assert PROMPTS_DIR.exists()
        assert PROMPTS_DIR.is_dir()

    def test_nexus_prompts_exist(self):
        """Test that all Nexus prompts exist."""
        nexus_dir = PROMPTS_DIR / "nexus"
        assert nexus_dir.exists()

        # Check required files
        assert (nexus_dir / "system_prompt.md").exists()
        assert (nexus_dir / "primitive_blocks_guide.md").exists()
        assert (nexus_dir / "graph_structure_guide.md").exists()

    def test_skills_dir_exists(self):
        """Test that the skills directory exists."""
        skills_dir = PROMPTS_DIR / "skills"
        assert skills_dir.exists()
        assert skills_dir.is_dir()


@pytest.mark.unit
class TestCacheManagement:
    """Tests for cache management."""

    def test_clear_prompt_cache(self):
        """Test that cache clearing works."""
        # Load some prompts to populate cache
        load_prompt("nexus", "system_prompt")
        get_skill_guide("gmail")

        # Clear cache
        clear_prompt_cache()

        # Check cache was cleared
        load_cache = load_prompt.cache_info()
        skill_cache = get_skill_guide.cache_info()

        assert load_cache.currsize == 0
        assert skill_cache.currsize == 0
