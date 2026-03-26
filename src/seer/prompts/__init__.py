"""
Prompt loading utilities for Seer agents and MCP resources.

This module provides centralized prompt management, loading markdown files
from the prompts directory and caching them for performance.

Usage:
    from seer.prompts import load_prompt, get_nexus_system_prompt

    # Load a specific prompt file
    blocks_guide = load_prompt("nexus", "primitive_blocks_guide")

    # Get the complete Nexus system prompt (base only, without dynamic content)
    system_prompt = get_nexus_system_prompt()

    # Get skill guides
    gmail_guide = get_skill_guide("gmail")
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

from seer.logger import get_logger

logger = get_logger(__name__)

# Base directory for prompts
PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=32)
def load_prompt(category: str, name: str) -> str:
    """
    Load a prompt from a markdown file.

    Args:
        category: The prompt category (e.g., "nexus", "skills")
        name: The prompt name without extension (e.g., "system_prompt", "gmail")

    Returns:
        The prompt content as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    prompt_path = PROMPTS_DIR / category / f"{name}.md"

    if not prompt_path.exists():
        logger.error("Prompt file not found: %s", prompt_path)
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    content = prompt_path.read_text(encoding="utf-8")
    logger.debug("Loaded prompt: %s/%s (%d chars)", category, name, len(content))
    return content


def get_nexus_system_prompt() -> str:
    """
    Get the base Nexus system prompt.

    This returns only the static base prompt without dynamic content like
    schema references, examples, or block guides. The agent should compose
    the full prompt by combining this with dynamic content.

    Returns:
        The base system prompt for the Nexus workflow assistant
    """
    return load_prompt("nexus", "system_prompt")


def get_primitive_blocks_guide() -> str:
    """
    Get the comprehensive guide for primitive workflow blocks.

    Documents all block types (tool, llm, mcp, if, for_each) with
    schemas, examples, and usage patterns.

    Returns:
        Formatted markdown guide for primitive blocks
    """
    return load_prompt("nexus", "primitive_blocks_guide")


def get_graph_structure_guide() -> str:
    """
    Get the guide for workflow graph structure and compilation.

    Documents how nodes and edges compile to LangGraph, edge types,
    diamond patterns, loop handling, and validation rules.

    Returns:
        Formatted markdown guide for graph structure
    """
    return load_prompt("nexus", "graph_structure_guide")


@lru_cache(maxsize=16)
def get_skill_guide(skill_name: str) -> Optional[str]:
    """
    Get the integration skill guide for a specific provider.

    Args:
        skill_name: The skill/integration name (e.g., "gmail", "slack")

    Returns:
        The skill guide content, or None if not found
    """
    try:
        return load_prompt("skills", skill_name.lower())
    except FileNotFoundError:
        logger.warning("Skill guide not found: %s", skill_name)
        return None


def list_available_skills() -> list[str]:
    """
    List all available skill guides.

    Returns:
        List of skill names that have guides available
    """
    skills_dir = PROMPTS_DIR / "skills"
    if not skills_dir.exists():
        return []

    return [f.stem for f in skills_dir.glob("*.md")]


def get_datetime_context(user_timezone: str | None = None) -> str:
    """Return current date/time string for injection into LLM system prompts."""
    now = datetime.now(timezone.utc)
    parts = [
        f"Current date and time: {now.strftime('%Y-%m-%d %H:%M')} UTC ({now.strftime('%A')})"
    ]
    if user_timezone:
        try:
            from zoneinfo import ZoneInfo  # pylint: disable=import-outside-toplevel  # Reason: Only needed when timezone provided

            local = now.astimezone(ZoneInfo(user_timezone))
            parts.append(
                f"User's local time: {local.strftime('%Y-%m-%d %H:%M')} {user_timezone}"
            )
        except (KeyError, Exception):  # pylint: disable=broad-exception-caught  # Reason: Invalid tz must not break prompt
            pass
    return "\n".join(parts)


def clear_prompt_cache() -> None:
    """
    Clear all cached prompts.

    Useful for development when prompt files are being modified.
    """
    load_prompt.cache_clear()
    get_skill_guide.cache_clear()
    logger.info("Prompt cache cleared")
