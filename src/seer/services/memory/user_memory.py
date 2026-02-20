"""
User memory service for cross-session context.

Wraps Mem0 operations with async interface and business logic
for user-scoped memory management.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from seer.config import config
from seer.logger import get_logger
from seer.services.memory.mem0_client import get_mem0_client

logger = get_logger(__name__)


class UserMemoryService:
    """
    Service for managing per-user memories.

    Provides async wrappers around Mem0's synchronous API, with additional
    business logic for formatting and filtering memories.

    Usage:
        service = UserMemoryService()
        await service.add_memory("user_123", "User prefers Slack notifications")
        memories = await service.search("user_123", "notification preferences")
    """

    def __init__(self):
        """Initialize the memory service."""
        self._client = None

    @property
    def client(self):
        """Lazy load the Mem0 client."""
        if self._client is None:
            self._client = get_mem0_client()
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if memory service is available (enabled and client initialized)."""
        return config.memory_enabled and self.client is not None

    async def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Add memory for a user.

        Mem0 automatically extracts facts from the content using LLM,
        so you can pass full conversation text or individual messages.

        Args:
            user_id: Unique user identifier (e.g., Clerk user_id)
            content: Content to extract memories from
            metadata: Optional metadata (session_id, workflow_id, etc.)

        Returns:
            Mem0 add result with extracted memories, or None if unavailable
        """
        if not self.is_available:
            logger.debug("Memory service unavailable, skipping add_memory")
            return None

        try:
            # Prepare metadata with timestamp
            mem_metadata = {
                "added_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            }

            # Run sync Mem0 operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.add(
                    content,
                    user_id=user_id,
                    metadata=mem_metadata,
                )
            )

            logger.debug(
                "Added memory for user %s: %d facts extracted",
                user_id,
                len(result.get("results", [])) if isinstance(result, dict) else 0,
            )
            return result

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Memory is non-critical, must not block main flow
            logger.warning("Failed to add memory for user %s: %s", user_id, e)
            return None

    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search user memories by semantic similarity.

        Args:
            user_id: Unique user identifier
            query: Search query (semantic search)
            limit: Maximum results to return
            filters: Optional metadata filters

        Returns:
            List of matching memories with scores
        """
        if not self.is_available:
            return []

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    query,
                    user_id=user_id,
                    limit=limit,
                )
            )

            memories = result.get("results", []) if isinstance(result, dict) else []

            # Apply additional filters if provided
            if filters:
                memories = self._apply_filters(memories, filters)

            logger.debug(
                "Found %d memories for user %s matching query: %s",
                len(memories),
                user_id,
                query[:50],
            )
            return memories

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Memory is non-critical
            logger.warning("Failed to search memories for user %s: %s", user_id, e)
            return []

    async def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories for a user.

        Useful for debugging or displaying user's memory profile.

        Args:
            user_id: Unique user identifier

        Returns:
            List of all user memories
        """
        if not self.is_available:
            return []

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.get_all(user_id=user_id)
            )

            memories = result.get("results", []) if isinstance(result, dict) else []
            logger.debug("Retrieved %d memories for user %s", len(memories), user_id)
            return memories

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Memory is non-critical
            logger.warning("Failed to get memories for user %s: %s", user_id, e)
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: Mem0 memory identifier

        Returns:
            True if deleted successfully
        """
        if not self.is_available:
            return False

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete(memory_id)
            )
            logger.debug("Deleted memory %s", memory_id)
            return True

        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Memory is non-critical
            logger.warning("Failed to delete memory %s: %s", memory_id, e)
            return False

    async def get_context_for_prompt(
        self,
        user_id: str,
        current_query: str,
        max_memories: Optional[int] = None,
    ) -> str:
        """
        Get formatted memory context for injection into agent system prompt.

        Searches for relevant memories and formats them for LLM consumption.

        Args:
            user_id: Unique user identifier
            current_query: Current user query for relevance search
            max_memories: Override for max memories (uses config default)

        Returns:
            Formatted string for system prompt injection, or empty string
        """
        if not self.is_available or not config.memory_context_injection_enabled:
            return ""

        limit = max_memories or config.memory_context_max_memories

        # If no query provided, get recent memories instead of semantic search
        if current_query:
            memories = await self.search(user_id, current_query, limit=limit)
        else:
            all_memories = await self.get_all(user_id)
            memories = all_memories[:limit]

        if not memories:
            return ""

        return self._format_memories_for_prompt(memories)

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories for system prompt injection.

        Creates a concise, structured format that gives the agent context
        about the user without overwhelming the prompt.
        """
        if not memories:
            return ""

        lines = ["## User Context (from memory)"]
        lines.append("The following facts are known about this user from previous sessions:\n")

        for memory in memories:
            # Extract memory text - Mem0 stores it in 'memory' field
            text = memory.get("memory", memory.get("text", str(memory)))
            # Truncate very long memories
            if len(text) > 200:
                text = text[:197] + "..."
            lines.append(f"- {text}")

        lines.append("")  # Empty line at end
        return "\n".join(lines)

    def _apply_filters(
        self,
        memories: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Apply metadata filters to memories.

        Supports:
        - has_session_id: Filter to only memories with session metadata
        - session_id: Filter to specific session
        - workflow_id: Filter to specific workflow
        """
        filtered = memories

        if filters.get("has_session_id"):
            filtered = [
                m for m in filtered
                if m.get("metadata", {}).get("session_id") is not None
            ]

        if "session_id" in filters:
            filtered = [
                m for m in filtered
                if m.get("metadata", {}).get("session_id") == filters["session_id"]
            ]

        if "workflow_id" in filters:
            filtered = [
                m for m in filtered
                if m.get("metadata", {}).get("workflow_id") == filters["workflow_id"]
            ]

        return filtered


# Convenience function for getting a service instance
_SERVICE_INSTANCE: Optional[UserMemoryService] = None


def get_user_memory_service() -> UserMemoryService:
    """Get or create the UserMemoryService singleton."""
    global _SERVICE_INSTANCE  # pylint: disable=global-statement  # Reason: Singleton pattern
    if _SERVICE_INSTANCE is None:
        _SERVICE_INSTANCE = UserMemoryService()
    return _SERVICE_INSTANCE
