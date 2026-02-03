"""Knowledge base workflow tools."""
from __future__ import annotations

from seer.logger import get_logger
from seer.tools.base import register_tool
from seer.tools.knowledge.add import KnowledgeBaseAddTextTool
from seer.tools.knowledge.common import KNOWLEDGE_BASE_PICKER
from seer.tools.knowledge.list import KnowledgeBaseListTool
from seer.tools.knowledge.query import KnowledgeBaseQueryTool

logger = get_logger("tools.knowledge")


def register_knowledge_tools() -> None:
    """Register all knowledge base tools."""
    register_tool(KnowledgeBaseQueryTool())
    register_tool(KnowledgeBaseAddTextTool())
    register_tool(KnowledgeBaseListTool())
    logger.debug("Registered knowledge base tools")


__all__ = [
    "KnowledgeBaseQueryTool",
    "KnowledgeBaseAddTextTool",
    "KnowledgeBaseListTool",
    "KNOWLEDGE_BASE_PICKER",
    "register_knowledge_tools",
]
