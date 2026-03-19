"""Shared helpers for workflow memory tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowMemoryAccess, WorkflowRuntimeContext


def require_memory_access(context: "WorkflowRuntimeContext | None") -> "WorkflowMemoryAccess":
    """Require a workflow runtime context with an injected memory adapter."""
    if context is None or context.memory_access is None:
        raise HTTPException(status_code=500, detail="Memory access is not available in this workflow runtime")
    return context.memory_access


__all__ = ["require_memory_access"]
