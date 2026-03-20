"""Workflow memory tools."""
from __future__ import annotations

from seer.logger import get_logger
from seer.tools.base import register_tool
from seer.tools.memory.add import MemoryAddTool
from seer.tools.memory.delete import MemoryDeleteTool
from seer.tools.memory.get import MemoryGetTool
from seer.tools.memory.search import MemorySearchTool
from seer.tools.memory.update import MemoryUpdateTool

logger = get_logger("tools.memory")


def register_memory_tools() -> None:
    """Register the workflow memory tool family."""
    register_tool(MemoryAddTool())
    register_tool(MemoryGetTool())
    register_tool(MemorySearchTool())
    register_tool(MemoryUpdateTool())
    register_tool(MemoryDeleteTool())
    logger.debug("Registered memory tools")


__all__ = [
    "MemoryAddTool",
    "MemoryDeleteTool",
    "MemoryGetTool",
    "MemorySearchTool",
    "MemoryUpdateTool",
    "register_memory_tools",
]
