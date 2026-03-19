"""
Compatibility wrapper for the legacy user-scoped memory API.

The public interface remains keyed by ``user_id`` so existing Nexus and API
callers keep working, but all bank-aware operations now route through the
default memory bank of the user's personal organization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from seer.database import User
from seer.logger import get_logger
from seer.services.memory.mem0_client import get_mem0_client
from seer.services.memory.memory_bank_service import MemoryBankMemoryService, MemoryBankService, MemoryNotFoundError

logger = get_logger(__name__)


class UserMemoryService:
    """Legacy compatibility API that delegates to the personal default bank."""

    def __init__(self):
        self._client = None
        self._bank_service = MemoryBankService()
        self._bank_memory_service = MemoryBankMemoryService()

    @property
    def client(self):
        """Expose the underlying Mem0 client for compatibility and tests."""
        if self._client is None:
            self._client = get_mem0_client()
            if self._client is not None:
                self._bank_memory_service._client = self._client  # pylint: disable=protected-access  # Reason: keep wrapper and bank service on the same mocked client in tests
        return self._client

    @property
    def is_available(self) -> bool:
        return self._bank_memory_service.is_available

    async def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.add_memory(user, bank, content=content, metadata=metadata, infer=infer)

    async def create_manual_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.create_manual_memory(user, bank, content, metadata=metadata)

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.search(user, bank, query=query, limit=limit, filters=filters)

    async def get_all(
        self,
        user_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.get_all(user, bank, filters=filters)

    async def get_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if user_id is None:
            if not self.is_available:
                return None
            try:
                return self.client.get(memory_id) if self.client is not None else None
            except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: compatibility lookup should fail softly
                logger.warning("Failed to get memory %s: %s", memory_id, exc)
                return None

        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.get_memory(user, bank, memory_id)

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        if user_id is None:
            existing_memory = await self.get_memory(memory_id)
            if existing_memory is None:
                return None
            user_id = existing_memory.get("user_id")
            if not user_id:
                return None

        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.update_memory(user, bank, memory_id, content=content, metadata=metadata)

    async def delete_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        if user_id is None:
            existing_memory = await self.get_memory(memory_id)
            if existing_memory is None:
                return False
            user_id = existing_memory.get("user_id")
            if not user_id:
                return False

        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.delete_memory(user, bank, memory_id)

    async def get_context_for_prompt(
        self,
        user_id: str,
        current_query: str,
        max_memories: Optional[int] = None,
    ) -> str:
        user, bank = await self._resolve_default_bank(user_id)
        return await self._bank_memory_service.get_context_for_prompt(
            user,
            bank,
            current_query=current_query,
            max_memories=max_memories,
        )

    async def _resolve_default_bank(self, user_id: str):
        user = await User.get_or_none(user_id=user_id)
        if user is None:
            raise MemoryNotFoundError(f"User not found for memory operations: {user_id}")
        bank = await self._bank_service.get_or_create_default_bank(user)
        return user, bank


_SERVICE_INSTANCE: Optional[UserMemoryService] = None


def get_user_memory_service() -> UserMemoryService:
    """Get or create the shared UserMemoryService instance."""
    global _SERVICE_INSTANCE  # pylint: disable=global-statement  # Reason: module-level singleton
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = UserMemoryService()
    return _SERVICE_INSTANCE


__all__ = ["UserMemoryService", "get_user_memory_service"]
