"""
Scratch storage tools package.

Provides tools for explicit agent-controlled scratch storage of large datasets.
The agent decides when to use scratch storage via fetch_to_scratch, keeping
existing data-fetching tools unchanged.
"""
from seer.tools.base import register_tool
from seer.tools.scratch.fetch_to_scratch import FetchToScratchTool


def register_scratch_tools() -> None:
    """Register all scratch storage tools."""
    register_tool(FetchToScratchTool())


__all__ = [
    "FetchToScratchTool",
    "register_scratch_tools",
]
