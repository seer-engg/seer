"""Database models for memory bank management."""
from __future__ import annotations

from enum import Enum

from tortoise import fields, models


MEMORY_BANK_ID_PREFIX = "mb_"


class MemoryBankStatus(str, Enum):
    """Lifecycle states for memory banks."""

    ACTIVE = "active"
    DELETED = "deleted"


class MemoryBank(models.Model):
    """Organization-owned logical partition for workflow and Nexus memory."""

    id = fields.IntField(primary_key=True)
    organization = fields.ForeignKeyField("models.Organization", related_name="memory_banks", on_delete=fields.CASCADE)
    organization_id: int  # Tortoise ORM FK shadow attribute
    created_by_user = fields.ForeignKeyField("models.User", related_name="created_memory_banks", on_delete=fields.CASCADE)
    created_by_user_id: int  # Tortoise ORM FK shadow attribute
    name = fields.CharField(max_length=255)
    description = fields.TextField(null=True)
    status = fields.CharEnumField(MemoryBankStatus, default=MemoryBankStatus.ACTIVE)
    is_default = fields.BooleanField(default=False)
    namespace_key = fields.CharField(max_length=255, unique=True)
    last_used_at = fields.DatetimeField(null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "memory_banks"
        unique_together = (("organization", "name"),)
        indexes = (
            ("organization_id", "status"),
            ("organization_id", "is_default"),
        )

    def __str__(self) -> str:
        return f"MemoryBank<{self.public_id}:{self.name}>"

    @property
    def public_id(self) -> str:
        """Return public ID with stable prefix."""
        return f"{MEMORY_BANK_ID_PREFIX}{self.id}"

    @classmethod
    def parse_public_id(cls, public_id: str) -> int:
        """Parse public memory bank ID into the internal integer key."""
        if not public_id.startswith(MEMORY_BANK_ID_PREFIX):
            raise ValueError(f"Invalid memory bank ID format: {public_id}")
        return int(public_id[len(MEMORY_BANK_ID_PREFIX):])


__all__ = [
    "MEMORY_BANK_ID_PREFIX",
    "MemoryBankStatus",
    "MemoryBank",
]
