"""Bank-aware Mem0 operations for organization-scoped memory banks."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional

from seer.config import config
from seer.database import MemoryBank, User
from seer.logger import get_logger
from seer.services.memory.mem0_client import get_mem0_client
from seer.services.memory.memory_bank_common import utcnow
from seer.services.organization_service import ensure_personal_organization

logger = get_logger(__name__)


class MemoryBankMemoryService:
    """Perform bank-aware memory operations against Mem0."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy load the shared Mem0 client."""
        if self._client is None:
            self._client = get_mem0_client()
        return self._client

    @property
    def is_available(self) -> bool:
        return config.memory_enabled and self.client is not None

    async def add_memory(
        self,
        user: User,
        bank: MemoryBank,
        *,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Add content to a specific bank."""
        if not self.is_available:
            logger.debug("Memory service unavailable, skipping add_memory")
            return None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.add(
                    content,
                    agent_id=bank.namespace_key,
                    metadata=self._build_metadata(bank, user, metadata),
                    infer=infer,
                ),
            )
            await self._touch_bank(bank)
            return result if isinstance(result, dict) else None
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: memory writes must not crash caller paths
            logger.warning("Failed to add memory to bank %s: %s", bank.public_id, exc)
            return None

    async def create_manual_memory(
        self,
        user: User,
        bank: MemoryBank,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store a single verbatim memory entry."""
        created = await self.add_memory(
            user,
            bank,
            content=content,
            metadata={"source": "manual", **(metadata or {})},
            infer=False,
        )
        if not isinstance(created, dict):
            return None
        results = created.get("results") or []
        if not results:
            return None
        memory_id = results[0].get("id")
        if not memory_id:
            return None
        return await self.get_memory(user, bank, memory_id)

    async def search(
        self,
        user: User,
        bank: MemoryBank,
        *,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search a bank using semantic retrieval plus bank filters."""
        if not self.is_available:
            return []

        mem0_filters = self._extract_mem0_filters(filters)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query,
                    agent_id=bank.namespace_key,
                    limit=limit,
                    filters=self._build_query_filters(bank, mem0_filters),
                ),
            )
            memories = result.get("results", []) if isinstance(result, dict) else []
            if await self._supports_legacy_user_namespace(bank, user):
                legacy_result = await loop.run_in_executor(
                    None,
                    lambda: self.client.search(
                        query,
                        user_id=user.user_id,
                        limit=limit,
                        filters=mem0_filters or None,
                    ),
                )
                memories = self._merge_memories(memories, legacy_result.get("results", []) if isinstance(legacy_result, dict) else [])
            await self._touch_bank(bank)
            return self._apply_filters(memories, filters)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: memory search is best-effort
            logger.warning("Failed to search memories for bank %s: %s", bank.public_id, exc)
            return []

    async def get_all(
        self,
        user: User,
        bank: MemoryBank,
        *,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List memories stored in a bank."""
        if not self.is_available:
            return []

        mem0_filters = self._extract_mem0_filters(filters)

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.get_all(
                    agent_id=bank.namespace_key,
                    filters=self._build_query_filters(bank, mem0_filters),
                    limit=limit,
                ),
            )
            memories = result.get("results", []) if isinstance(result, dict) else result or []
            if await self._supports_legacy_user_namespace(bank, user):
                legacy_result = await loop.run_in_executor(
                    None,
                    lambda: self.client.get_all(
                        user_id=user.user_id,
                        filters=mem0_filters or None,
                        limit=limit,
                    ),
                )
                legacy_memories = legacy_result.get("results", []) if isinstance(legacy_result, dict) else legacy_result or []
                memories = self._merge_memories(memories, legacy_memories)
            return self._apply_filters(memories, filters)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: memory listing is best-effort
            logger.warning("Failed to list memories for bank %s: %s", bank.public_id, exc)
            return []

    async def get_memory(
        self,
        user: User,
        bank: MemoryBank,
        memory_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a single memory and ensure it belongs to the requested bank."""
        if not self.is_available:
            return None

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.client.get(memory_id))
            if not isinstance(result, dict):
                return None
            if not await self._memory_belongs_to_bank(result, bank, user):
                return None
            return result
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: direct memory lookup is best-effort
            logger.warning("Failed to get memory %s from bank %s: %s", memory_id, bank.public_id, exc)
            return None

    async def update_memory(
        self,
        user: User,
        bank: MemoryBank,
        memory_id: str,
        *,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update a memory while preserving bank metadata and legacy compatibility."""
        if not self.is_available:
            return None

        existing_memory = await self.get_memory(user, bank, memory_id)
        if existing_memory is None:
            return None

        preserved_metadata = {
            **(existing_memory.get("metadata") or {}),
            **self._build_metadata(bank, user, metadata, include_timestamps=False),
            "edited_at": utcnow().isoformat(),
        }

        try:
            loop = asyncio.get_event_loop()

            def _update() -> None:
                existing_embeddings = {
                    content: self.client.embedding_model.embed(content, "update")
                }
                self.client._update_memory(  # pylint: disable=protected-access  # Reason: Mem0 public update drops custom metadata
                    memory_id,
                    content,
                    existing_embeddings,
                    metadata=preserved_metadata,
                )

            await loop.run_in_executor(None, _update)
            await self._touch_bank(bank)
            return await self.get_memory(user, bank, memory_id)
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: memory updates should fail softly
            logger.warning("Failed to update memory %s in bank %s: %s", memory_id, bank.public_id, exc)
            return None

    async def delete_memory(
        self,
        user: User,
        bank: MemoryBank,
        memory_id: str,
    ) -> bool:
        """Delete a memory after verifying bank ownership."""
        if not self.is_available:
            return False

        existing_memory = await self.get_memory(user, bank, memory_id)
        if existing_memory is None:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.client.delete(memory_id))
            await self._touch_bank(bank)
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: delete is best-effort
            logger.warning("Failed to delete memory %s in bank %s: %s", memory_id, bank.public_id, exc)
            return False

    async def get_context_for_prompt(
        self,
        user: User,
        bank: MemoryBank,
        *,
        current_query: str,
        max_memories: Optional[int] = None,
    ) -> str:
        """Build prompt context for an attached memory bank."""
        if not self.is_available or not config.memory_context_injection_enabled:
            return ""

        limit = max_memories or config.memory_context_max_memories
        if current_query:
            memories = await self.search(user, bank, query=current_query, limit=limit)
        else:
            memories = (await self.get_all(user, bank, limit=limit))[:limit]

        if not memories:
            return ""

        return self._format_memories_for_prompt(memories)

    def _build_metadata(
        self,
        bank: MemoryBank,
        user: User,
        metadata: Optional[Dict[str, Any]],
        *,
        include_timestamps: bool = True,
    ) -> Dict[str, Any]:
        bank_metadata = {
            "organization_id": bank.organization_id,
            "memory_bank_id": bank.public_id,
            "memory_bank_namespace": bank.namespace_key,
            "created_by_user_id": user.user_id,
        }
        if include_timestamps:
            bank_metadata["added_at"] = utcnow().isoformat()
        if metadata:
            bank_metadata.update(metadata)
        return bank_metadata

    def _build_query_filters(self, bank: MemoryBank, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "organization_id": bank.organization_id,
            "memory_bank_id": bank.public_id,
            "memory_bank_namespace": bank.namespace_key,
            **(filters or {}),
        }

    def _extract_mem0_filters(self, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not filters:
            return {}
        return {
            key: value
            for key, value in filters.items()
            if key in {"session_id", "workflow_id", "source"}
        }

    async def _touch_bank(self, bank: MemoryBank) -> None:
        bank.last_used_at = utcnow()
        await bank.save(update_fields=["last_used_at"])

    async def _supports_legacy_user_namespace(self, bank: MemoryBank, user: User) -> bool:
        if not bank.is_default:
            return False
        personal_org, _ = await ensure_personal_organization(user)
        return bank.organization_id == personal_org.id

    async def _memory_belongs_to_bank(self, memory: Dict[str, Any], bank: MemoryBank, user: User) -> bool:
        metadata = memory.get("metadata") or {}
        if metadata.get("memory_bank_id") == bank.public_id:
            return True
        if metadata.get("memory_bank_namespace") == bank.namespace_key:
            return True
        return await self._supports_legacy_user_namespace(bank, user) and memory.get("user_id") == user.user_id

    def _merge_memories(self, primary: Iterable[Dict[str, Any]], secondary: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: list[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for memory in list(primary) + list(secondary):
            memory_id = memory.get("id")
            if memory_id and memory_id in seen_ids:
                continue
            if memory_id:
                seen_ids.add(memory_id)
            merged.append(memory)
        return merged

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""

        lines = ["## User Context (from memory)", "The following facts are known from previous sessions:\n"]
        for memory in memories:
            text = memory.get("memory", memory.get("text", str(memory)))
            if len(text) > 200:
                text = text[:197] + "..."
            lines.append(f"- {text}")

        lines.append("")
        return "\n".join(lines)

    def _apply_filters(self, memories: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not filters:
            return memories

        filtered = memories
        if filters.get("has_session_id"):
            filtered = [memory for memory in filtered if memory.get("metadata", {}).get("session_id") is not None]
        if "session_id" in filters:
            filtered = [memory for memory in filtered if memory.get("metadata", {}).get("session_id") == filters["session_id"]]
        if "workflow_id" in filters:
            filtered = [memory for memory in filtered if memory.get("metadata", {}).get("workflow_id") == filters["workflow_id"]]
        if "source" in filters:
            filtered = [memory for memory in filtered if memory.get("metadata", {}).get("source") == filters["source"]]
        return filtered


__all__ = ["MemoryBankMemoryService"]
