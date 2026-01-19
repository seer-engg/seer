"""
Container for shared compiler dependencies (registries, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.schema_registry import SchemaRegistry


@dataclass(frozen=True)
class CompilerContext:
    schema_registry: SchemaRegistry
    tool_registry: ToolRegistry
    model_registry: ModelRegistry
