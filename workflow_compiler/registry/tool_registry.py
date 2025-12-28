"""
Simple in-memory tool registry abstraction.

The compiler references this registry to determine the JSON schema of tool
outputs (for type-safety) and to locate the callable that should run at
execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, MutableMapping, Optional

from workflow_compiler.schema.models import JsonSchema


ToolCallable = Callable[[Dict[str, Any], Dict[str, Any] | None], Any]


@dataclass
class ToolDefinition:
    name: str
    version: str
    input_schema: JsonSchema
    output_schema: JsonSchema
    handler: ToolCallable


class ToolNotFoundError(KeyError):
    """Raised when attempting to access an unknown tool."""


class ToolRegistry:
    """
    Stores tool implementations and their schemas.
    """

    def __init__(self, initial: MutableMapping[str, ToolDefinition] | None = None) -> None:
        self._tools: Dict[str, ToolDefinition] = dict(initial or {})

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool '{name}' is not registered") from exc

    def maybe_get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)


