"""
Notion tools - search, read/write pages, query and create database entries.

All Notion tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.notion.search import NotionSearchTool
from seer.tools.notion.pages import (
    NotionGetPageTool,
    NotionCreatePageTool,
    NotionUpdatePageTool,
    NotionGetPageContentTool,
    NotionAppendPageContentTool,
)
from seer.tools.notion.databases import (
    NotionQueryDatabaseTool,
    NotionCreateDatabaseEntryTool,
)


def register_notion_tools():
    """Register all Notion tools with the tool registry."""
    register_tool(NotionSearchTool())
    register_tool(NotionGetPageTool())
    register_tool(NotionCreatePageTool())
    register_tool(NotionUpdatePageTool())
    register_tool(NotionGetPageContentTool())
    register_tool(NotionAppendPageContentTool())
    register_tool(NotionQueryDatabaseTool())
    register_tool(NotionCreateDatabaseEntryTool())


__all__ = [
    "register_notion_tools",
    "NotionSearchTool",
    "NotionGetPageTool",
    "NotionCreatePageTool",
    "NotionUpdatePageTool",
    "NotionGetPageContentTool",
    "NotionAppendPageContentTool",
    "NotionQueryDatabaseTool",
    "NotionCreateDatabaseEntryTool",
]
