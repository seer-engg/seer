from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from seer.database import Organization, User

if TYPE_CHECKING:
    from seer.core.files.service import WorkflowFileSystem
    from seer.core.scratch.store import ScratchStore


class WorkflowMemoryAccess(Protocol):
    """Runtime protocol implemented by services-side memory adapters."""

    async def get_prompt_context(self, memory_bank_id: str, current_query: str, max_memories: int | None = None) -> str: ...

    async def search(self, memory_bank_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]: ...

    async def add(
        self,
        memory_bank_id: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
        infer: bool = True,
    ) -> Dict[str, Any] | None: ...

    async def get(self, memory_bank_id: str, memory_id: str) -> Dict[str, Any] | None: ...

    async def update(
        self,
        memory_bank_id: str,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None: ...

    async def delete(self, memory_bank_id: str, memory_id: str) -> bool: ...


@dataclass
class WorkflowRunBudget:
    """Mutable run-level cost tracking state."""

    per_run_cost_cap_usd: float | None = None
    accumulated_cost_usd: float = 0.0


@dataclass
class WorkflowRuntimeResources:
    """Lazy-loaded runtime resources resolved on demand."""

    file_system: Optional["WorkflowFileSystem"] = None
    scratch_store: Optional["ScratchStore"] = None
    organization: Optional[Organization] = None
    organization_loaded: bool = False


@dataclass
class WorkflowRuntimeContext:  # pylint: disable=too-many-instance-attributes  # Reason: attributes are already grouped into sub-dataclasses; splitting further would hurt readability
    """
    Carries runtime-scoped data that needs to be accessible to LangGraph
    nodes and tool handlers. Extend this as new fields are required.

    Attributes:
        user: The user executing the workflow.
        workflow_run_id: The run ID (e.g., "run_123") for file storage scoping.
        thread_id: LangGraph thread ID for chat contexts.
        organization_id: The organization ID for resolving shared OAuth connections.
        memory_access: Optional services-side adapter for bank-aware memory access.
    """

    user: User
    workflow_run_id: str | None = None
    thread_id: str | None = None  # For chat threads
    organization_id: int | None = None  # For shared OAuth connection resolution
    memory_access: WorkflowMemoryAccess | None = None
    budget: WorkflowRunBudget = field(default_factory=WorkflowRunBudget)
    tool_bindings: Dict[str, Dict[str, Any]] | None = None  # Per-tool credential bindings from workflow spec

    # Private field for lazy-loaded runtime resources
    _resources: WorkflowRuntimeResources = field(default_factory=WorkflowRuntimeResources, repr=False)

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments  # Reason: preserve existing constructor contract while storing budget internally
        self,
        user: User,
        workflow_run_id: str | None = None,
        thread_id: str | None = None,
        per_run_cost_cap_usd: float | None = None,
        accumulated_cost_usd: float = 0.0,
        organization_id: int | None = None,
        memory_access: WorkflowMemoryAccess | None = None,
    ) -> None:
        self.user = user
        self.workflow_run_id = workflow_run_id
        self.thread_id = thread_id
        self.organization_id = organization_id
        self.memory_access = memory_access
        self.budget = WorkflowRunBudget(
            per_run_cost_cap_usd=per_run_cost_cap_usd,
            accumulated_cost_usd=accumulated_cost_usd,
        )
        self.tool_bindings = None
        self._resources = WorkflowRuntimeResources()

    @property
    def per_run_cost_cap_usd(self) -> float | None:
        """Return the configured run-level cost cap."""
        return self.budget.per_run_cost_cap_usd

    @per_run_cost_cap_usd.setter
    def per_run_cost_cap_usd(self, value: float | None) -> None:
        self.budget.per_run_cost_cap_usd = value

    @property
    def accumulated_cost_usd(self) -> float:
        """Return the accumulated run cost."""
        return self.budget.accumulated_cost_usd

    @accumulated_cost_usd.setter
    def accumulated_cost_usd(self, value: float) -> None:
        self.budget.accumulated_cost_usd = value

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
        if self._resources.file_system is None:
            # pylint: disable=import-outside-toplevel  # Avoid circular imports with file service
            from seer.core.files.service import WorkflowFileSystem
            self._resources.file_system = WorkflowFileSystem.instance()
        return self._resources.file_system

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

    @property
    def scratch_store(self) -> "ScratchStore":
        """
        Get the scratch data store instance.

        The scratch store is lazily loaded on first access. It shares the
        same S3 backend as the file system but with a different path prefix.

        Returns:
            ScratchStore singleton instance.

        Raises:
            ValueError: If S3 storage is not configured.
        """
        if self._resources.scratch_store is None:
            # pylint: disable=import-outside-toplevel  # Avoid circular imports with scratch store
            from seer.core.scratch.store import ScratchStore
            self._resources.scratch_store = ScratchStore.instance()
        return self._resources.scratch_store

    @property
    def has_scratch_store(self) -> bool:
        """
        Check if the scratch store is available.

        Returns:
            True if S3 storage is configured (same requirement as file system).
        """
        return self.has_file_system

    async def get_organization(self) -> "Organization | None":
        """Resolve and cache the runtime organization when one is available."""
        if self.organization_id is None:
            return None

        if self._resources.organization_loaded:
            return self._resources.organization

        self._resources.organization = await Organization.get_or_none(id=self.organization_id)
        self._resources.organization_loaded = True
        return self._resources.organization
