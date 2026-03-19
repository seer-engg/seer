"""Workflow memory get tool."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.tools.base import BaseTool
from seer.tools.memory.base import require_memory_access

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials


class MemoryGetTool(BaseTool):
    """Fetch a specific memory by ID from a bank."""

    name = "memory_get"
    description = "Fetch a specific memory from a memory bank."
    integration_type = "memory"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_bank_id": {"type": "string", "description": "Target memory bank ID (format: mb_*)"},
                "memory_id": {"type": "string", "description": "Memory ID to fetch"},
            },
            "required": ["memory_bank_id", "memory_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Any:
        _ = access_token, credentials
        memory_access = require_memory_access(context)
        return await memory_access.get(arguments["memory_bank_id"], arguments["memory_id"])


__all__ = ["MemoryGetTool"]
