"""
Memory API endpoints for user memory management.

Allows users to view, search, and delete their memories stored by the
Mem0 memory layer. Useful for:
- Transparency: Users can see what the agent "knows" about them
- Privacy: Users can delete specific memories or all memories (GDPR)
- Debugging: Admins can inspect memory extraction results
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from seer.config import config
from seer.logger import get_logger
from seer.services.memory import UserMemoryService

logger = get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


# ============================================================================
# Response Models
# ============================================================================


class MemoryItem(BaseModel):
    """A single memory item."""

    id: str = Field(description="Unique memory identifier")
    memory: str = Field(description="The memory content/fact")
    score: Optional[float] = Field(default=None, description="Relevance score (for search results)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Associated metadata")
    created_at: Optional[str] = Field(default=None, description="When the memory was created")


class MemoryListResponse(BaseModel):
    """Response for listing memories."""

    memories: List[MemoryItem]
    total: int
    user_id: str


class MemorySearchResponse(BaseModel):
    """Response for searching memories."""

    query: str
    memories: List[MemoryItem]
    total: int


class MemoryDeleteResponse(BaseModel):
    """Response for deleting memories."""

    deleted: bool
    memory_id: Optional[str] = None
    message: str


class MemoryStatsResponse(BaseModel):
    """Response for memory statistics."""

    total_memories: int
    memory_enabled: bool
    extraction_enabled: bool
    injection_enabled: bool


# ============================================================================
# Helper Functions
# ============================================================================


def _format_memory_item(raw_memory: Dict[str, Any]) -> MemoryItem:
    """Convert raw Mem0 memory to API response format."""
    return MemoryItem(
        id=raw_memory.get("id", ""),
        memory=raw_memory.get("memory", raw_memory.get("text", str(raw_memory))),
        score=raw_memory.get("score"),
        metadata=raw_memory.get("metadata"),
        created_at=raw_memory.get("created_at") or raw_memory.get("metadata", {}).get("added_at"),
    )


def _get_user_id_from_request(request: Request) -> str:
    """Extract user_id from the authenticated request."""
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user.user_id


def _check_memory_enabled():
    """Raise 503 if memory feature is disabled."""
    if not config.memory_enabled:
        raise HTTPException(
            status_code=503,
            detail="Memory feature is not enabled. Set MEMORY_ENABLED=true to enable."
        )


# ============================================================================
# Endpoints
# ============================================================================


@router.get("", response_model=MemoryListResponse)
async def list_memories(request: Request) -> MemoryListResponse:
    """
    List all memories for the authenticated user.

    Returns all stored facts and context that the agent has learned
    about the user from previous sessions.
    """
    _check_memory_enabled()
    user_id = _get_user_id_from_request(request)

    memory_service = UserMemoryService()
    memories = await memory_service.get_all(user_id=user_id)

    return MemoryListResponse(
        memories=[_format_memory_item(m) for m in memories],
        total=len(memories),
        user_id=user_id,
    )


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
) -> MemorySearchResponse:
    """
    Search user memories by semantic similarity.

    Use this to find specific memories or facts. The search is semantic,
    so it will find related concepts even if exact words don't match.
    """
    _check_memory_enabled()
    user_id = _get_user_id_from_request(request)

    memory_service = UserMemoryService()
    memories = await memory_service.search(
        user_id=user_id,
        query=q,
        limit=limit,
    )

    return MemorySearchResponse(
        query=q,
        memories=[_format_memory_item(m) for m in memories],
        total=len(memories),
    )


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(
    request: Request,
    memory_id: str,
) -> MemoryDeleteResponse:
    """
    Delete a specific memory by ID.

    Use this to remove incorrect or unwanted memories. The agent will
    no longer have access to this information.
    """
    _check_memory_enabled()
    user_id = _get_user_id_from_request(request)

    # Verify the memory belongs to this user by searching for it first
    memory_service = UserMemoryService()
    all_memories = await memory_service.get_all(user_id=user_id)
    memory_ids = {m.get("id") for m in all_memories}

    if memory_id not in memory_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Memory {memory_id} not found or does not belong to you"
        )

    success = await memory_service.delete_memory(memory_id)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory")

    logger.info("User %s deleted memory %s", user_id, memory_id)

    return MemoryDeleteResponse(
        deleted=True,
        memory_id=memory_id,
        message="Memory deleted successfully",
    )


@router.delete("", response_model=MemoryDeleteResponse)
async def delete_all_memories(request: Request) -> MemoryDeleteResponse:
    """
    Delete ALL memories for the authenticated user.

    WARNING: This action is irreversible. All learned context about
    the user will be permanently removed. Use for GDPR compliance
    or when the user wants a fresh start.
    """
    _check_memory_enabled()
    user_id = _get_user_id_from_request(request)

    memory_service = UserMemoryService()
    all_memories = await memory_service.get_all(user_id=user_id)

    if not all_memories:
        return MemoryDeleteResponse(
            deleted=True,
            message="No memories to delete",
        )

    # Delete each memory
    deleted_count = 0
    for memory in all_memories:
        memory_id = memory.get("id")
        if memory_id:
            success = await memory_service.delete_memory(memory_id)
            if success:
                deleted_count += 1

    logger.info("User %s deleted all memories (%d items)", user_id, deleted_count)

    return MemoryDeleteResponse(
        deleted=True,
        message=f"Deleted {deleted_count} memories",
    )


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request) -> MemoryStatsResponse:
    """
    Get memory statistics and feature status.

    Returns the total number of memories and whether various memory
    features are enabled.
    """
    user_id = _get_user_id_from_request(request)

    total_memories = 0
    if config.memory_enabled:
        memory_service = UserMemoryService()
        all_memories = await memory_service.get_all(user_id=user_id)
        total_memories = len(all_memories)

    return MemoryStatsResponse(
        total_memories=total_memories,
        memory_enabled=config.memory_enabled,
        extraction_enabled=config.memory_enabled and config.memory_extraction_enabled,
        injection_enabled=config.memory_enabled and config.memory_context_injection_enabled,
    )
