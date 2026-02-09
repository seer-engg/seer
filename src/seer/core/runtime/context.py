from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from seer.database import User

if TYPE_CHECKING:
    from seer.core.files.service import WorkflowFileSystem


@dataclass
class WorkflowRuntimeContext:
    """
    Carries runtime-scoped data that needs to be accessible to LangGraph
    nodes and tool handlers. Extend this as new fields are required.

    Attributes:
        user: The user executing the workflow.
        workflow_run_id: The run ID (e.g., "run_123") for file storage scoping.
        thread_id: LangGraph thread ID for chat contexts.
        per_run_cost_cap_usd: Maximum cost allowed for this run.
        accumulated_cost_usd: Running total of costs (mutable).
    """

    user: User
    workflow_run_id: str | None = None
    thread_id: str | None = None  # For chat threads
    per_run_cost_cap_usd: float | None = None  # Cost limit per execution
    accumulated_cost_usd: float = 0.0  # Running total (mutable)

    # Private field for lazy-loaded file system
    _file_system: Optional["WorkflowFileSystem"] = field(default=None, repr=False)

    @property
    def file_system(self) -> "WorkflowFileSystem":
        """
        Get the workflow file system instance.

        The file system is lazily loaded on first access to avoid import
        cycles and unnecessary initialization.

        Returns:
            WorkflowFileSystem singleton instance.

        Raises:
            ValueError: If workflow file storage is not configured.
        """
        if self._file_system is None:
            # pylint: disable=import-outside-toplevel  # Avoid circular imports with file service
            from seer.core.files.service import WorkflowFileSystem
            self._file_system = WorkflowFileSystem.instance()
        return self._file_system

    @property
    def has_file_system(self) -> bool:
        """
        Check if the workflow file system is available.

        Returns:
            True if file storage is configured, False otherwise.
        """
        # pylint: disable=import-outside-toplevel  # Avoid circular imports with config
        from seer.config import config
        return config.is_workflow_file_system_configured
