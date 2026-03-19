"""
Memory service for cross-session user context using Mem0.

This module provides:
- UserMemoryService: backward-compatible API keyed by user_id
- MemoryBankService: org-scoped bank CRUD and default-bank resolution
- MemoryBankMemoryService: bank-aware Mem0 operations
- WorkflowMemoryRuntimeAdapter: runtime-facing adapter for workflow execution
- get_mem0_client: Singleton Mem0 client factory

Memory Types:
- User Facts: Long-term preferences, patterns, and context (e.g., "User prefers Slack notifications")
- Session Summaries: Key decisions and outcomes from each conversation
- Episodic Search: Semantic search over past conversations

Usage:
    from seer.services.memory import UserMemoryService

    memory_service = UserMemoryService()
    await memory_service.add_memory(user_id="user_123", content="User prefers Slack...")
    memories = await memory_service.search(user_id="user_123", query="notification preferences")
"""

from seer.services.memory.mem0_client import get_mem0_client
from seer.services.memory.memory_bank_service import (
    MemoryAccessError,
    MemoryBankMemoryService,
    MemoryBankService,
    MemoryNotFoundError,
    MemoryServiceError,
    MemoryValidationError,
)
from seer.services.memory.runtime_adapter import WorkflowMemoryRuntimeAdapter
from seer.services.memory.user_memory import UserMemoryService

__all__ = [
    "get_mem0_client",
    "MemoryAccessError",
    "MemoryBankMemoryService",
    "MemoryBankService",
    "MemoryNotFoundError",
    "MemoryServiceError",
    "MemoryValidationError",
    "UserMemoryService",
    "WorkflowMemoryRuntimeAdapter",
]
