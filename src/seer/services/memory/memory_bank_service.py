"""Organization-scoped memory bank CRUD and bank resolution services."""

from __future__ import annotations

from typing import List, Optional
from uuid import uuid4

from tortoise.exceptions import IntegrityError

from seer.database import MemoryBank, MemoryBankStatus, Organization, OrganizationMembership, User
from seer.database.organization_models import MembershipStatus, OrganizationRole
from seer.services.memory.memory_bank_common import (
    DEFAULT_MEMORY_BANK_DESCRIPTION,
    DEFAULT_MEMORY_BANK_NAME,
    MemoryAccessError,
    MemoryBankResolution,
    MemoryNotFoundError,
    MemoryServiceError,
    MemoryValidationError,
    namespace_for_default_bank,
    namespace_for_named_bank,
)
from seer.services.memory.memory_bank_memory_service import MemoryBankMemoryService
from seer.services.organization_service import ensure_personal_organization


class MemoryBankService:
    """Manage organization-owned memory bank records."""

    async def resolve_organization(
        self,
        user: User,
        organization: Optional[Organization] = None,
        organization_id: Optional[int] = None,
    ) -> tuple[Organization, Optional[OrganizationMembership]]:
        """Resolve an organization and validate the user's active membership."""
        if organization is None:
            if organization_id is not None:
                organization = await Organization.get_or_none(id=organization_id)
                if organization is None:
                    raise MemoryNotFoundError(f"Organization {organization_id} not found")
            else:
                organization, membership = await ensure_personal_organization(user)
                return organization, membership

        membership = await OrganizationMembership.get_or_none(
            organization=organization,
            user=user,
            status=MembershipStatus.ACTIVE,
        )
        if membership is None:
            raise MemoryAccessError(f"User {user.user_id} does not have access to organization {organization.id}")
        return organization, membership

    async def list_banks(
        self,
        user: User,
        organization: Optional[Organization] = None,
        organization_id: Optional[int] = None,
    ) -> List[MemoryBank]:
        """List active banks for the resolved organization."""
        resolved_org, _ = await self.resolve_organization(user, organization=organization, organization_id=organization_id)
        await self.get_or_create_default_bank(user, resolved_org)
        return await MemoryBank.filter(organization=resolved_org, status=MemoryBankStatus.ACTIVE).order_by("name", "id")

    async def create_bank(
        self,
        user: User,
        organization: Optional[Organization],
        name: str,
        description: Optional[str] = None,
    ) -> MemoryBank:
        """Create a named bank in the given organization."""
        resolved_org, _ = await self.resolve_organization(user, organization=organization)
        normalized_name = name.strip()
        if not normalized_name:
            raise MemoryValidationError("Memory bank name cannot be empty")

        try:
            bank = await MemoryBank.create(
                organization=resolved_org,
                created_by_user=user,
                name=normalized_name,
                description=(description or "").strip() or None,
                namespace_key=f"pending:{uuid4()}",
            )
        except IntegrityError as exc:
            raise MemoryValidationError(f"Memory bank '{normalized_name}' already exists") from exc

        bank.namespace_key = namespace_for_named_bank(resolved_org.id, bank.public_id)
        await bank.save(update_fields=["namespace_key"])
        return bank

    async def get_bank_for_org(
        self,
        user: User,
        organization: Optional[Organization],
        bank_id: str,
    ) -> MemoryBank:
        """Resolve a bank by public id within an accessible organization."""
        resolved_org, _ = await self.resolve_organization(user, organization=organization)
        return await self._get_bank_for_org(resolved_org, bank_id)

    async def _get_bank_for_org(self, organization: Organization, bank_id: str) -> MemoryBank:
        try:
            internal_id = MemoryBank.parse_public_id(bank_id)
        except ValueError as exc:
            raise MemoryValidationError(str(exc)) from exc

        bank = await MemoryBank.get_or_none(
            id=internal_id,
            organization=organization,
            status=MemoryBankStatus.ACTIVE,
        )
        if bank is None:
            raise MemoryNotFoundError(f"Memory bank not found: {bank_id}")
        return bank

    async def get_default_bank(self, organization: Organization) -> Optional[MemoryBank]:
        """Return the active default bank for an organization, if one exists."""
        return await MemoryBank.get_or_none(
            organization=organization,
            status=MemoryBankStatus.ACTIVE,
            is_default=True,
        )

    async def get_or_create_default_bank(
        self,
        user: User,
        organization: Optional[Organization] = None,
        organization_id: Optional[int] = None,
    ) -> MemoryBank:
        """Get or lazily create the default bank for an organization."""
        resolved_org, _ = await self.resolve_organization(user, organization=organization, organization_id=organization_id)
        existing = await self.get_default_bank(resolved_org)
        if existing is not None:
            return existing

        try:
            return await MemoryBank.create(
                organization=resolved_org,
                created_by_user=user,
                name=DEFAULT_MEMORY_BANK_NAME,
                description=DEFAULT_MEMORY_BANK_DESCRIPTION,
                is_default=True,
                namespace_key=namespace_for_default_bank(resolved_org.id),
            )
        except IntegrityError:
            existing = await self.get_default_bank(resolved_org)
            if existing is None:
                raise
            return existing

    async def update_bank(
        self,
        user: User,
        organization: Optional[Organization],
        bank_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> MemoryBank:
        """Update a bank's editable metadata."""
        bank = await self.get_bank_for_org(user, organization, bank_id)
        updates: list[str] = []

        if name is not None:
            normalized_name = name.strip()
            if not normalized_name:
                raise MemoryValidationError("Memory bank name cannot be empty")
            bank.name = normalized_name
            updates.append("name")

        if description is not None:
            bank.description = description.strip() or None
            updates.append("description")

        if updates:
            try:
                await bank.save(update_fields=updates)
            except IntegrityError as exc:
                raise MemoryValidationError(f"Memory bank '{bank.name}' already exists") from exc
        return bank

    async def set_default_bank(
        self,
        user: User,
        organization: Optional[Organization],
        bank_id: str,
    ) -> MemoryBank:
        """Promote a bank to be the organization's default bank."""
        resolved_org, membership = await self.resolve_organization(user, organization=organization)
        self._require_admin_or_owner(membership)

        bank = await self._get_bank_for_org(resolved_org, bank_id)
        await MemoryBank.filter(organization=resolved_org, status=MemoryBankStatus.ACTIVE, is_default=True).update(is_default=False)
        bank.is_default = True
        await bank.save(update_fields=["is_default"])
        return bank

    async def delete_bank(
        self,
        user: User,
        organization: Optional[Organization],
        bank_id: str,
    ) -> MemoryBank:
        """Soft-delete a non-default bank."""
        resolved_org, membership = await self.resolve_organization(user, organization=organization)
        self._require_admin_or_owner(membership)

        bank = await self._get_bank_for_org(resolved_org, bank_id)
        if bank.is_default:
            raise MemoryValidationError("Default memory bank cannot be deleted")

        bank.status = MemoryBankStatus.DELETED
        await bank.save(update_fields=["status"])
        return bank

    def _require_admin_or_owner(self, membership: Optional[OrganizationMembership]) -> None:
        if membership is None or membership.role not in (OrganizationRole.OWNER, OrganizationRole.ADMIN):
            raise MemoryAccessError("Only organization admins can perform this memory bank operation")


__all__ = [
    "DEFAULT_MEMORY_BANK_DESCRIPTION",
    "DEFAULT_MEMORY_BANK_NAME",
    "MemoryAccessError",
    "MemoryBankMemoryService",
    "MemoryBankResolution",
    "MemoryBankService",
    "MemoryNotFoundError",
    "MemoryServiceError",
    "MemoryValidationError",
]
