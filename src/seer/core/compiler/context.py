"""
Container for shared compiler dependencies (registries, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from seer.core.registry.mcp_client_registry import MCPClientRegistry
from seer.core.registry.model_registry import ModelRegistry
from seer.core.registry.tool_registry import ToolRegistry
from seer.core.schema.schema_registry import SchemaRegistry


@dataclass(frozen=True)
class CompilerContext:
    schema_registry: SchemaRegistry
    tool_registry: ToolRegistry
    model_registry: ModelRegistry
    mcp_client_registry: MCPClientRegistry

    # Auto-resolved connection IDs for nodes in single-account scenarios.
    # Maps node_id -> connection_id. Set by validate_connections during compilation
    # when user has exactly one connection for a provider.
    resolved_connections: Dict[str, int] = field(default_factory=dict)
