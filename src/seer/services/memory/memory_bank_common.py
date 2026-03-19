"""Shared types and helpers for organization-scoped memory bank services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from seer.core.errors import ExecutionError
from seer.database import MemoryBank, Organization, OrganizationMembership

DEFAULT_MEMORY_BANK_NAME = "Default"
DEFAULT_MEMORY_BANK_DESCRIPTION = "Default workspace memory bank"


class MemoryServiceError(ExecutionError):
    """Base exception for memory-bank operations."""


class MemoryAccessError(MemoryServiceError):
    """Raised when a user is not allowed to access a memory bank."""


class MemoryNotFoundError(MemoryServiceError):
    """Raised when a memory bank or memory item does not exist."""


class MemoryValidationError(MemoryServiceError):
    """Raised when a memory operation is invalid."""


def utcnow() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def namespace_for_default_bank(organization_id: int) -> str:
    """Build the namespace used by the default bank for an organization."""
    return f"org:{organization_id}:default"


def namespace_for_named_bank(organization_id: int, public_id: str) -> str:
    """Build the namespace used by a named bank for an organization."""
    return f"org:{organization_id}:bank:{public_id}"


@dataclass(slots=True)
class MemoryBankResolution:
    """Resolved organization and bank pair for runtime operations."""

    organization: Organization
    membership: Optional[OrganizationMembership]
    bank: MemoryBank


__all__ = [
    "DEFAULT_MEMORY_BANK_DESCRIPTION",
    "DEFAULT_MEMORY_BANK_NAME",
    "MemoryAccessError",
    "MemoryBankResolution",
    "MemoryNotFoundError",
    "MemoryServiceError",
    "MemoryValidationError",
    "namespace_for_default_bank",
    "namespace_for_named_bank",
    "utcnow",
]
