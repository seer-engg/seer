"""
Airtable tools - list bases/tables, CRUD operations on records.

All Airtable tools are registered here for easy import and initialization.
"""

from seer.tools.base import register_tool
from seer.tools.airtable.bases import (
    AirtableListBasesTool,
    AirtableListTablesTool,
)
from seer.tools.airtable.records import (
    AirtableListRecordsTool,
    AirtableCreateRecordTool,
    AirtableUpdateRecordTool,
    AirtableDeleteRecordTool,
)


def register_airtable_tools():
    """Register all Airtable tools with the tool registry."""
    register_tool(AirtableListBasesTool())
    register_tool(AirtableListTablesTool())
    register_tool(AirtableListRecordsTool())
    register_tool(AirtableCreateRecordTool())
    register_tool(AirtableUpdateRecordTool())
    register_tool(AirtableDeleteRecordTool())


__all__ = [
    "register_airtable_tools",
    "AirtableListBasesTool",
    "AirtableListTablesTool",
    "AirtableListRecordsTool",
    "AirtableCreateRecordTool",
    "AirtableUpdateRecordTool",
    "AirtableDeleteRecordTool",
]
