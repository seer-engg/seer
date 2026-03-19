"""
Memory API endpoints for organization-scoped memory bank management.

This router keeps the legacy ``/api/memory`` surface working by mapping it to
the caller's default bank in the active organization, while also exposing
explicit bank management and bank-scoped CRUD routes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from seer.api.core.middleware.organization import get_membership, get_organization
from seer.config import config
from seer.database import MemoryBank, Organization, OrganizationMembership, User
from seer.logger import get_logger
from seer.services.memory import (
    MemoryAccessError,
    MemoryBankMemoryService,
    MemoryBankService,
    MemoryNotFoundError,
    MemoryValidationError,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryItem(BaseModel):
    """A single memory item."""

    id: str = Field(description="Unique memory identifier")
    memory: str = Field(description="The memory content/fact")
    score: Optional[float] = Field(default=None, description="Relevance score for search results")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Associated metadata")
    created_at: Optional[str] = Field(default=None, description="When the memory was created")
    updated_at: Optional[str] = Field(default=None, description="When the memory was last updated")


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


class MemoryMutationResponse(BaseModel):
    """Response for creating or updating a memory."""

    memory: MemoryItem
    message: str


class MemoryStatsResponse(BaseModel):
    """Response for memory statistics."""

    total_memories: int
    memory_enabled: bool
    extraction_enabled: bool
    injection_enabled: bool


class MemoryUpsertRequest(BaseModel):
    """Request body for creating or updating a memory."""

    memory: str = Field(..., min_length=1, max_length=2000, description="Memory content")


class MemoryBankItem(BaseModel):
    """Memory bank summary used by bank management endpoints."""

    memory_bank_id: str
    name: str
    description: Optional[str] = None
    is_default: bool
    memory_count: int
    created_at: str
    updated_at: str


class MemoryBankListResponse(BaseModel):
    """Response for listing memory banks."""

    items: List[MemoryBankItem]
    total: int


class MemoryBankMutationRequest(BaseModel):
    """Request payload for creating or updating a memory bank."""

    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)


class MemoryBankUpdateRequest(BaseModel):
    """Request payload for patching an existing memory bank."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=2000)


def _format_memory_item(raw_memory: Dict[str, Any]) -> MemoryItem:
    """Convert raw Mem0 memory to API response format."""
    metadata = raw_memory.get("metadata") or {}
    return MemoryItem(
        id=raw_memory.get("id", ""),
        memory=raw_memory.get("memory", raw_memory.get("text", str(raw_memory))),
        score=raw_memory.get("score"),
        metadata=metadata,
        created_at=raw_memory.get("created_at") or metadata.get("added_at"),
        updated_at=raw_memory.get("updated_at") or metadata.get("edited_at"),
    )


def _check_memory_enabled() -> None:
    """Raise 503 if memory is disabled."""
    if not config.memory_enabled:
        raise HTTPException(status_code=503, detail="Memory feature is not enabled. Set MEMORY_ENABLED=true to enable.")


def _normalize_memory_content(content: str) -> str:
    """Normalize and validate user-submitted memory content."""
    normalized = content.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="Memory content cannot be empty")
    return normalized


def _require_db_user(request: Request) -> User:
    """Get the authenticated database user from request state."""
    user = getattr(request.state, "db_user", None)
    if user is None:
        user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user


def _get_user_id(user: User) -> str:
    user_id = getattr(user, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return user_id


def _get_org_context(request: Request) -> Tuple[Optional[Organization], Optional[OrganizationMembership]]:
    """Get active organization context from middleware when available."""
    try:
        org = get_organization(request)
        membership = get_membership(request)
        return org, membership
    except Exception:  # pylint: disable=broad-exception-caught  # Reason: request-level fallback to personal org when org middleware is absent
        return None, None


def _raise_memory_error(exc: Exception) -> None:
    """Map memory service errors onto HTTP responses."""
    if isinstance(exc, MemoryNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, MemoryAccessError):
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if isinstance(exc, MemoryValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _get_bank_item(
    bank: MemoryBank,
    user: User,
    memory_service: MemoryBankMemoryService,
) -> MemoryBankItem:
    """Hydrate a bank summary with current memory count."""
    await bank.fetch_related("organization")
    memories = await memory_service.get_all(user, bank)
    return MemoryBankItem(
        memory_bank_id=bank.public_id,
        name=bank.name,
        description=bank.description,
        is_default=bank.is_default,
        memory_count=len(memories),
        created_at=bank.created_at.isoformat(),
        updated_at=bank.updated_at.isoformat(),
    )


async def _resolve_bank_from_request(request: Request, bank_id: str) -> tuple[User, MemoryBank]:
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    try:
        bank = await bank_service.get_bank_for_org(user, org, bank_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)
    return user, bank


async def _resolve_default_bank_from_request(request: Request) -> tuple[User, MemoryBank]:
    """Resolve the active organization's default memory bank for compatibility routes."""
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    try:
        bank = await bank_service.get_or_create_default_bank(user, organization=org)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)
    return user, bank


@router.get("/banks", response_model=MemoryBankListResponse)
async def list_memory_banks(request: Request) -> MemoryBankListResponse:
    """List active memory banks in the caller's current organization."""
    _check_memory_enabled()
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    bank_memory_service = MemoryBankMemoryService()

    try:
        banks = await bank_service.list_banks(user, organization=org)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)

    items = [await _get_bank_item(bank, user, bank_memory_service) for bank in banks]
    return MemoryBankListResponse(items=items, total=len(items))


@router.post("/banks", response_model=MemoryBankItem, status_code=status.HTTP_201_CREATED)
async def create_memory_bank(request: Request, payload: MemoryBankMutationRequest) -> MemoryBankItem:
    """Create a new memory bank in the current organization."""
    _check_memory_enabled()
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    bank_memory_service = MemoryBankMemoryService()

    try:
        bank = await bank_service.create_bank(user, org, payload.name, description=payload.description)
        await bank.fetch_related("organization")
        return await _get_bank_item(bank, user, bank_memory_service)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)


@router.get("/banks/{bank_id}", response_model=MemoryBankItem)
async def get_memory_bank(request: Request, bank_id: str) -> MemoryBankItem:
    """Get a single memory bank by ID."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    bank_memory_service = MemoryBankMemoryService()
    return await _get_bank_item(bank, user, bank_memory_service)


@router.patch("/banks/{bank_id}", response_model=MemoryBankItem)
async def update_memory_bank(request: Request, bank_id: str, payload: MemoryBankUpdateRequest) -> MemoryBankItem:
    """Patch a memory bank."""
    _check_memory_enabled()
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    bank_memory_service = MemoryBankMemoryService()

    try:
        bank = await bank_service.update_bank(
            user,
            org,
            bank_id,
            name=payload.name,
            description=payload.description,
        )
        await bank.fetch_related("organization")
        return await _get_bank_item(bank, user, bank_memory_service)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)


@router.delete("/banks/{bank_id}", response_model=MemoryDeleteResponse)
async def delete_memory_bank(request: Request, bank_id: str) -> MemoryDeleteResponse:
    """Soft-delete a non-default memory bank."""
    _check_memory_enabled()
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()

    try:
        await bank_service.delete_bank(user, org, bank_id)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)

    return MemoryDeleteResponse(deleted=True, memory_id=bank_id, message="Memory bank deleted successfully")


@router.post("/banks/{bank_id}/set-default", response_model=MemoryBankItem)
async def set_default_memory_bank(request: Request, bank_id: str) -> MemoryBankItem:
    """Promote a memory bank to be the default bank in the current organization."""
    _check_memory_enabled()
    user = _require_db_user(request)
    org, _ = _get_org_context(request)
    bank_service = MemoryBankService()
    bank_memory_service = MemoryBankMemoryService()

    try:
        bank = await bank_service.set_default_bank(user, org, bank_id)
        await bank.fetch_related("organization")
        return await _get_bank_item(bank, user, bank_memory_service)
    except Exception as exc:  # pylint: disable=broad-exception-caught  # Reason: route-level normalization into HTTP responses
        _raise_memory_error(exc)


@router.get("/banks/{bank_id}/items", response_model=MemoryListResponse)
async def list_bank_memories(request: Request, bank_id: str) -> MemoryListResponse:
    """List all memories in a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    bank_memory_service = MemoryBankMemoryService()
    memories = await bank_memory_service.get_all(user, bank)
    return MemoryListResponse(memories=[_format_memory_item(memory) for memory in memories], total=len(memories), user_id=_get_user_id(user))


@router.get("/banks/{bank_id}/items/search", response_model=MemorySearchResponse)
async def search_bank_memories(
    request: Request,
    bank_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
) -> MemorySearchResponse:
    """Search memories in a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    bank_memory_service = MemoryBankMemoryService()
    memories = await bank_memory_service.search(user, bank, query=q, limit=limit)
    return MemorySearchResponse(query=q, memories=[_format_memory_item(memory) for memory in memories], total=len(memories))


@router.post("/banks/{bank_id}/items", response_model=MemoryMutationResponse)
async def create_bank_memory(
    request: Request,
    bank_id: str,
    payload: MemoryUpsertRequest,
) -> MemoryMutationResponse:
    """Create a manual memory inside a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    content = _normalize_memory_content(payload.memory)
    bank_memory_service = MemoryBankMemoryService()
    created = await bank_memory_service.create_manual_memory(user, bank, content)
    if created is None:
        raise HTTPException(status_code=500, detail="Failed to create memory")
    return MemoryMutationResponse(memory=_format_memory_item(created), message="Memory created successfully")


@router.get("/banks/{bank_id}/items/{memory_id}", response_model=MemoryItem)
async def get_bank_memory(request: Request, bank_id: str, memory_id: str) -> MemoryItem:
    """Get a single memory in a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    bank_memory_service = MemoryBankMemoryService()
    memory = await bank_memory_service.get_memory(user, bank, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found in bank {bank_id}")
    return _format_memory_item(memory)


@router.put("/banks/{bank_id}/items/{memory_id}", response_model=MemoryMutationResponse)
async def update_bank_memory(
    request: Request,
    bank_id: str,
    memory_id: str,
    payload: MemoryUpsertRequest,
) -> MemoryMutationResponse:
    """Update a memory in a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    content = _normalize_memory_content(payload.memory)
    bank_memory_service = MemoryBankMemoryService()
    updated = await bank_memory_service.update_memory(user, bank, memory_id, content=content)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found in bank {bank_id}")
    return MemoryMutationResponse(memory=_format_memory_item(updated), message="Memory updated successfully")


@router.delete("/banks/{bank_id}/items/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_bank_memory(request: Request, bank_id: str, memory_id: str) -> MemoryDeleteResponse:
    """Delete a memory in a specific bank."""
    _check_memory_enabled()
    user, bank = await _resolve_bank_from_request(request, bank_id)
    bank_memory_service = MemoryBankMemoryService()
    deleted = await bank_memory_service.delete_memory(user, bank, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found in bank {bank_id}")
    return MemoryDeleteResponse(deleted=True, memory_id=memory_id, message="Memory deleted successfully")


@router.get("", response_model=MemoryListResponse)
async def list_memories(request: Request) -> MemoryListResponse:
    """List memories from the caller's default bank in the active organization."""
    _check_memory_enabled()
    user, bank = await _resolve_default_bank_from_request(request)
    memories = await MemoryBankMemoryService().get_all(user, bank)
    return MemoryListResponse(memories=[_format_memory_item(memory) for memory in memories], total=len(memories), user_id=_get_user_id(user))


@router.get("/search", response_model=MemorySearchResponse)
async def search_memories(
    request: Request,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
) -> MemorySearchResponse:
    """Search the caller's default bank in the active organization."""
    _check_memory_enabled()
    user, bank = await _resolve_default_bank_from_request(request)
    memories = await MemoryBankMemoryService().search(user, bank, query=q, limit=limit)
    return MemorySearchResponse(query=q, memories=[_format_memory_item(memory) for memory in memories], total=len(memories))


@router.post("", response_model=MemoryMutationResponse)
async def create_memory(request: Request, payload: MemoryUpsertRequest) -> MemoryMutationResponse:
    """Create a manual memory in the caller's default bank."""
    _check_memory_enabled()
    content = _normalize_memory_content(payload.memory)
    user, bank = await _resolve_default_bank_from_request(request)
    created = await MemoryBankMemoryService().create_manual_memory(user, bank, content)
    if created is None:
        raise HTTPException(status_code=500, detail="Failed to create memory")
    return MemoryMutationResponse(memory=_format_memory_item(created), message="Memory created successfully")


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats(request: Request) -> MemoryStatsResponse:
    """Get memory feature status and counts for the caller's default bank."""
    user, bank = await _resolve_default_bank_from_request(request)

    total_memories = 0
    if config.memory_enabled:
        total_memories = len(await MemoryBankMemoryService().get_all(user, bank))

    return MemoryStatsResponse(
        total_memories=total_memories,
        memory_enabled=config.memory_enabled,
        extraction_enabled=config.memory_enabled and config.memory_extraction_enabled,
        injection_enabled=config.memory_enabled and config.memory_context_injection_enabled,
    )


@router.put("/{memory_id}", response_model=MemoryMutationResponse)
async def update_memory(request: Request, memory_id: str, payload: MemoryUpsertRequest) -> MemoryMutationResponse:
    """Update a memory in the caller's default bank."""
    _check_memory_enabled()
    content = _normalize_memory_content(payload.memory)
    user, bank = await _resolve_default_bank_from_request(request)
    memory_service = MemoryBankMemoryService()
    existing = await memory_service.get_memory(user, bank, memory_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found or does not belong to you")
    updated = await memory_service.update_memory(
        user,
        bank,
        memory_id,
        content=content,
        metadata={"source": (existing.get("metadata") or {}).get("source", "manual")},
    )
    if updated is None:
        raise HTTPException(status_code=500, detail="Failed to update memory")
    return MemoryMutationResponse(memory=_format_memory_item(updated), message="Memory updated successfully")


@router.delete("/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(request: Request, memory_id: str) -> MemoryDeleteResponse:
    """Delete a memory from the caller's default bank."""
    _check_memory_enabled()
    user, bank = await _resolve_default_bank_from_request(request)
    memory_service = MemoryBankMemoryService()
    existing = await memory_service.get_memory(user, bank, memory_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found or does not belong to you")
    deleted = await memory_service.delete_memory(user, bank, memory_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete memory")
    return MemoryDeleteResponse(deleted=True, memory_id=memory_id, message="Memory deleted successfully")


@router.delete("", response_model=MemoryDeleteResponse)
async def delete_all_memories(request: Request) -> MemoryDeleteResponse:
    """Delete all memories from the caller's default bank."""
    _check_memory_enabled()
    user, bank = await _resolve_default_bank_from_request(request)
    memory_service = MemoryBankMemoryService()
    all_memories = await memory_service.get_all(user, bank)

    deleted_count = 0
    for memory in all_memories:
        memory_id = memory.get("id")
        if memory_id and await memory_service.delete_memory(user, bank, memory_id):
            deleted_count += 1

    return MemoryDeleteResponse(deleted=True, message=f"Deleted {deleted_count} memories")


__all__ = ["router"]
