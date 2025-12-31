"""
Container for shared compiler dependencies (registries, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from workflow_compiler.registry.model_registry import ModelRegistry
from workflow_compiler.registry.tool_registry import ToolRegistry
from workflow_compiler.schema.schema_registry import SchemaRegistry


@dataclass(frozen=True)
class CompilerContext:
    schema_registry: SchemaRegistry
    tool_registry: ToolRegistry
    model_registry: ModelRegistry


