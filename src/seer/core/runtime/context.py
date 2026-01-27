from __future__ import annotations

from dataclasses import dataclass

from seer.database import User


@dataclass
class WorkflowRuntimeContext:
    """
    Carries runtime-scoped data that needs to be accessible to LangGraph
    nodes and tool handlers. Extend this as new fields are required.
    """

    user: User
    workflow_run_id: str | None = None
    thread_id: str | None = None  # For chat threads
    per_run_cost_cap_usd: float | None = None  # Cost limit per execution
    accumulated_cost_usd: float = 0.0  # Running total (mutable)
