"""
Simple in-memory tool registry abstraction.

The compiler references this registry to determine the JSON schema of tool
outputs (for type-safety) and to locate the callable that should run at
execution time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
)

from seer.core.schema.models import JsonSchema

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext


ToolCallable = Callable[
    [Dict[str, Any], Dict[str, Any] | None, "WorkflowRuntimeContext | None"],
    Any,
]
ToolAsyncCallable = Callable[
    [Dict[str, Any], Dict[str, Any] | None, "WorkflowRuntimeContext | None"],
    Awaitable[Any],
]


@dataclass
class ToolDefinition:
    name: str
    version: str
    input_schema: JsonSchema
    output_schema: JsonSchema
    handler: ToolCallable
    async_handler: ToolAsyncCallable | None = None


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

    def all(self) -> List[ToolDefinition]:
        return list(self._tools.values())
