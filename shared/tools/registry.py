"""
Tool metadata registry.
Manages tool entries and metadata.
"""
from dataclasses import dataclass
from typing import Dict


@dataclass
class ToolEntry:
    """Metadata for a single MCP tool."""
    name: str
    description: str
    service: str
