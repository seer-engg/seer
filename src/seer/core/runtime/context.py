from __future__ import annotations

from dataclasses import dataclass

from seer.database import User


@dataclass(frozen=True)
class WorkflowRuntimeContext:
    """
    Carries runtime-scoped data that needs to be accessible to LangGraph
    nodes and tool handlers. Extend this as new fields are required.
    """

    user: User
    workflow_run_id: str | None = None
