"""Workflow runtime adapter for memory bank access."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from seer.core.errors import ExecutionError
from seer.database import MemoryBank, Organization, User
from seer.logger import get_logger
from seer.services.memory.memory_bank_service import (
    MemoryAccessError,
    MemoryBankMemoryService,
    MemoryBankService,
    MemoryNotFoundError,
    MemoryValidationError,
)

logger = get_logger(__name__)


class WorkflowMemoryRuntimeAdapter:
    """Services-side adapter injected into WorkflowRuntimeContext."""

    def __init__(
        self,
        *,
        user: User,
        organization_id: Optional[int] = None,
        bank_service: Optional[MemoryBankService] = None,
        bank_memory_service: Optional[MemoryBankMemoryService] = None,
    ):
        self.user = user
        self.organization_id = organization_id
        self.bank_service = bank_service or MemoryBankService()
        self.bank_memory_service = bank_memory_service or MemoryBankMemoryService()

    async def get_prompt_context(self, memory_bank_id: str, current_query: str, max_memories: Optional[int] = None) -> str:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.get_context_for_prompt(self.user, bank, current_query=current_query, max_memories=max_memories)

    async def search(self, memory_bank_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.search(self.user, bank, query=query, limit=limit)

    async def add(
        self,
        memory_bank_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.add_memory(self.user, bank, content=content, metadata=metadata, infer=infer)

    async def get(self, memory_bank_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.get_memory(self.user, bank, memory_id)

    async def update(
        self,
        memory_bank_id: str,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.update_memory(self.user, bank, memory_id, content=content, metadata=metadata)

    async def delete(self, memory_bank_id: str, memory_id: str) -> bool:
        bank = await self._resolve_bank(memory_bank_id)
        return await self.bank_memory_service.delete_memory(self.user, bank, memory_id)

    async def _resolve_bank(self, memory_bank_id: str) -> MemoryBank:
        try:
            organization = None
            if self.organization_id is not None:
                organization = await Organization.get_or_none(id=self.organization_id)
                if organization is None:
                    raise MemoryNotFoundError(f"Organization {self.organization_id} not found")
            return await self.bank_service.get_bank_for_org(self.user, organization, memory_bank_id)
        except (MemoryNotFoundError, MemoryAccessError, MemoryValidationError) as exc:
            raise ExecutionError(str(exc)) from exc
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: normalize unexpected adapter failures for runtime callers
            logger.exception("Unexpected workflow memory adapter failure")
            raise ExecutionError(f"Memory access failed: {exc}") from exc


__all__ = ["WorkflowMemoryRuntimeAdapter"]
