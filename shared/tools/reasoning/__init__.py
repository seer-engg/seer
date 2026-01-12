"""Reasoning tools for recursive and complex analysis."""

from shared.tools.base import register_tool
from shared.tools.reasoning.rlm_tool import RecursiveLanguageModelTool


def register_reasoning_tools():
    """Register reasoning tools in the global tool registry."""
    register_tool(RecursiveLanguageModelTool())


__all__ = [
    "register_reasoning_tools",
    "RecursiveLanguageModelTool",
]
