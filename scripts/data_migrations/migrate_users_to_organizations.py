#!/usr/bin/env python
"""
Migrate existing users to the organization support system.

This script creates personal organizations for all existing users and updates
related tables to set organization_id. It is idempotent and safe to run
multiple times.

Usage:
    uv run scripts/migrate_users_to_organizations.py [--dry-run] [--update-clerk]

Tables migrated:
    - organizations (creates personal org for each user)
    - organization_memberships (creates owner membership)
    - billing_profiles (links existing or creates new)
    - workflows (sets organization_id)
    - workflow_files (sets organization_id)
    - knowledge_bases (sets organization_id)
    - llm_usage_records (sets organization_id)
    - usage_counters (sets organization_id)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM
from seer.database.models import User
from seer.database.organization_models import (
    MembershipStatus,
    Organization,
    OrganizationMembership,
    OrganizationRole,
    OrganizationType,
)
from seer.database.subscription_models import BillingProfile, BillingProfileType
from seer.database.workflow_models import Workflow, WorkflowFile
from seer.database.knowledge_models import KnowledgeBase
from seer.database.usage_models import LLMUsageRecord, UsageCounter
from seer.services.organization_service import generate_slug


@dataclass
class MigrationStats:
    """Track migration statistics."""

    users_processed: int = 0
    organizations_created: int = 0
    organizations_existed: int = 0
    memberships_created: int = 0
    memberships_existed: int = 0
    billing_profiles_linked: int = 0
    billing_profiles_created: int = 0
    billing_profiles_already_linked: int = 0
    workflows_updated: int = 0
    workflow_files_updated: int = 0
    knowledge_bases_updated: int = 0
    llm_usage_records_updated: int = 0
    usage_counters_updated: int = 0
    clerk_updates: int = 0
    clerk_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Migrate existing users to organization support system")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database",
    )
    parser.add_argument(
        "--update-clerk",
        action="store_true",
        help="Update Clerk metadata with active_organization_id",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Migrate only a specific user by Clerk user_id (for testing)",
    )
    return parser.parse_args()


async def ensure_personal_org_exists(
    user: User,
    stats: MigrationStats,
    dry_run: bool,
) -> Organization | None:
    """
    Ensure user has a personal organization, creating if needed.

    Returns the personal organization or None if dry_run and org doesn't exist.
    """
    # Check for existing personal org
    existing_org = await Organization.get_or_none(
        owner=user,
        type=OrganizationType.PERSONAL,
    )

    if existing_org:
        stats.organizations_existed += 1
        return existing_org

    if dry_run:
        print(f"    [DRY RUN] Would create personal organization for user {user.user_id}")
        stats.organizations_created += 1
        return None

    # Generate organization name
    name_parts = []
    if user.first_name:
        name_parts.append(user.first_name)
    if not name_parts and user.email:
        name_parts.append(user.email.split("@")[0])
    if not name_parts:
        name_parts.append("Personal")

    org_name = f"{' '.join(name_parts)}'s Workspace"
    slug = generate_slug(f"personal-{user.user_id}")

    organization = await Organization.create(
        name=org_name,
        slug=slug,
        type=OrganizationType.PERSONAL,
        owner=user,
        settings={},
    )

    stats.organizations_created += 1
    print(f"    Created organization: {organization.name} (id={organization.id})")

    return organization


async def ensure_membership_exists(
    user: User,
    organization: Organization,
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Ensure user has owner membership in their personal org."""
    existing_membership = await OrganizationMembership.get_or_none(
        organization=organization,
        user=user,
    )

    if existing_membership:
        stats.memberships_existed += 1
        return

    if dry_run:
        print(f"    [DRY RUN] Would create membership for user {user.user_id}")
        stats.memberships_created += 1
        return

    await OrganizationMembership.create(
        organization=organization,
        user=user,
        role=OrganizationRole.OWNER,
        status=MembershipStatus.ACTIVE,
        joined_at=datetime.now(timezone.utc),
    )

    stats.memberships_created += 1
    print(f"    Created membership for user {user.user_id}")


async def ensure_billing_profile(
    user: User,
    organization: Organization | None,
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """
    Ensure billing profile exists and is linked to organization.

    Handles three cases:
    1. Existing profile with owner_organization_id set - skip
    2. Existing profile without owner_organization_id - link it
    3. No existing profile - create new one linked to org
    """
    # Check for existing billing profile for this user
    existing_profile = await BillingProfile.get_or_none(owner_user_id=user.id)

    if existing_profile:
        if existing_profile.owner_organization_id:
            # Already linked, nothing to do
            stats.billing_profiles_already_linked += 1
            return

        # Link existing profile to organization
        if dry_run:
            print(f"    [DRY RUN] Would link billing profile {existing_profile.id} to organization")
            stats.billing_profiles_linked += 1
            return

        if organization:
            existing_profile.owner_organization_id = organization.id
            await existing_profile.save(update_fields=["owner_organization_id"])
            stats.billing_profiles_linked += 1
            print(f"    Linked billing profile {existing_profile.id} to organization")
        return

    # No existing profile, create one
    if dry_run:
        print(f"    [DRY RUN] Would create billing profile for user {user.user_id}")
        stats.billing_profiles_created += 1
        return

    if organization:
        await BillingProfile.create(
            type=BillingProfileType.INDIVIDUAL,
            owner_user=user,
            owner_organization=organization,
        )
        stats.billing_profiles_created += 1
        print(f"    Created billing profile for user {user.user_id}")


async def update_user_resources(
    user: User,
    organization: Organization,
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Update all user resources to set organization_id."""

    # Workflows
    workflow_count = await Workflow.filter(
        user=user,
        organization_id=None,
    ).count()

    if workflow_count > 0:
        if dry_run:
            print(f"    [DRY RUN] Would update {workflow_count} workflows")
        else:
            await Workflow.filter(
                user=user,
                organization_id=None,
            ).update(organization_id=organization.id)
            print(f"    Updated {workflow_count} workflows")
        stats.workflows_updated += workflow_count

    # WorkflowFiles
    file_count = await WorkflowFile.filter(
        user=user,
        organization_id=None,
    ).count()

    if file_count > 0:
        if dry_run:
            print(f"    [DRY RUN] Would update {file_count} workflow files")
        else:
            await WorkflowFile.filter(
                user=user,
                organization_id=None,
            ).update(organization_id=organization.id)
            print(f"    Updated {file_count} workflow files")
        stats.workflow_files_updated += file_count

    # KnowledgeBases
    kb_count = await KnowledgeBase.filter(
        user=user,
        organization_id=None,
    ).count()

    if kb_count > 0:
        if dry_run:
            print(f"    [DRY RUN] Would update {kb_count} knowledge bases")
        else:
            await KnowledgeBase.filter(
                user=user,
                organization_id=None,
            ).update(organization_id=organization.id)
            print(f"    Updated {kb_count} knowledge bases")
        stats.knowledge_bases_updated += kb_count

    # LLMUsageRecords
    llm_count = await LLMUsageRecord.filter(
        user=user,
        organization_id=None,
    ).count()

    if llm_count > 0:
        if dry_run:
            print(f"    [DRY RUN] Would update {llm_count} LLM usage records")
        else:
            await LLMUsageRecord.filter(
                user=user,
                organization_id=None,
            ).update(organization_id=organization.id)
            print(f"    Updated {llm_count} LLM usage records")
        stats.llm_usage_records_updated += llm_count

    # UsageCounters
    counter_count = await UsageCounter.filter(
        user=user,
        organization_id=None,
    ).count()

    if counter_count > 0:
        if dry_run:
            print(f"    [DRY RUN] Would update {counter_count} usage counters")
        else:
            await UsageCounter.filter(
                user=user,
                organization_id=None,
            ).update(organization_id=organization.id)
            print(f"    Updated {counter_count} usage counters")
        stats.usage_counters_updated += counter_count


async def update_clerk_metadata(
    user: User,
    organization: Organization,
    stats: MigrationStats,
    dry_run: bool,
) -> None:
    """Update Clerk metadata with active_organization_id."""
    # Import here to avoid circular imports and allow script to run without Clerk
    try:
        from seer.services.clerk_service import set_active_organization
    except ImportError:
        print("    Warning: Could not import clerk_service, skipping Clerk update")
        stats.clerk_skipped += 1
        return

    if dry_run:
        print(f"    [DRY RUN] Would set Clerk active_organization_id to {organization.id}")
        stats.clerk_updates += 1
        return

    try:
        await set_active_organization(user.user_id, organization.id)
        stats.clerk_updates += 1
        print(f"    Updated Clerk metadata with org {organization.id}")
    except Exception as e:
        print(f"    Warning: Failed to update Clerk metadata: {e}")
        stats.clerk_skipped += 1


async def migrate_user(
    user: User,
    stats: MigrationStats,
    dry_run: bool,
    update_clerk: bool,
) -> None:
    """Migrate a single user to organization support."""
    print(f"  Processing user: {user.user_id} (id={user.id})")

    # Step 1: Ensure personal organization exists
    organization = await ensure_personal_org_exists(user, stats, dry_run)

    # In dry-run mode without existing org, we can't do resource updates
    # but we still want to count what would be created
    if organization is None and dry_run:
        # Check if there's an existing org we can use for counting
        existing_org = await Organization.get_or_none(
            owner=user,
            type=OrganizationType.PERSONAL,
        )
        if existing_org is None:
            # Truly new org in dry-run, skip resource updates but record billing profile
            await ensure_billing_profile(user, None, stats, dry_run)
            stats.users_processed += 1
            return
        organization = existing_org

    # Step 2: Ensure membership exists
    await ensure_membership_exists(user, organization, stats, dry_run)

    # Step 3: Handle billing profile
    await ensure_billing_profile(user, organization, stats, dry_run)

    # Step 4: Update all user resources
    await update_user_resources(user, organization, stats, dry_run)

    # Step 5: Optionally update Clerk metadata
    if update_clerk:
        await update_clerk_metadata(user, organization, stats, dry_run)

    stats.users_processed += 1


async def migrate() -> None:
    """Main migration function."""
    args = parse_args()
    dry_run = args.dry_run
    update_clerk = args.update_clerk
    user_filter = args.user_id

    await Tortoise.init(config=TORTOISE_ORM)

    mode_label = "[DRY RUN] " if dry_run else ""
    print(f"{mode_label}Starting migration: Users to Organization Support")
    print()

    stats = MigrationStats()

    # Build user query
    user_query = User.all()
    if user_filter:
        user_query = user_query.filter(user_id=user_filter)
        print(f"Filtering to user: {user_filter}")
        print()

    # Count total users for progress
    total_users = await user_query.count()
    print(f"Total users to process: {total_users}")
    print()

    # Process each user
    async for user in user_query:
        try:
            await migrate_user(user, stats, dry_run, update_clerk)
        except Exception as e:
            error_msg = f"Error migrating user {user.user_id}: {e}"
            print(f"  ERROR: {error_msg}")
            stats.errors.append(error_msg)

    # Print summary
    print()
    print("=" * 60)
    print(f"{mode_label}Migration Summary")
    print("=" * 60)
    print(f"  Users processed:              {stats.users_processed}")
    print()
    print("Organizations:")
    print(f"  Created:                      {stats.organizations_created}")
    print(f"  Already existed:              {stats.organizations_existed}")
    print()
    print("Memberships:")
    print(f"  Created:                      {stats.memberships_created}")
    print(f"  Already existed:              {stats.memberships_existed}")
    print()
    print("Billing Profiles:")
    print(f"  Created:                      {stats.billing_profiles_created}")
    print(f"  Linked to org:                {stats.billing_profiles_linked}")
    print(f"  Already linked:               {stats.billing_profiles_already_linked}")
    print()
    print("Resources Updated:")
    print(f"  Workflows:                    {stats.workflows_updated}")
    print(f"  Workflow files:               {stats.workflow_files_updated}")
    print(f"  Knowledge bases:              {stats.knowledge_bases_updated}")
    print(f"  LLM usage records:            {stats.llm_usage_records_updated}")
    print(f"  Usage counters:               {stats.usage_counters_updated}")

    if update_clerk:
        print()
        print("Clerk Updates:")
        print(f"  Successful:                   {stats.clerk_updates}")
        print(f"  Skipped/Failed:               {stats.clerk_skipped}")

    if stats.errors:
        print()
        print("ERRORS:")
        for error in stats.errors:
            print(f"  - {error}")

    if dry_run:
        print()
        print("(No changes were written - re-run without --dry-run to apply)")

    await Tortoise.close_connections()


if __name__ == "__main__":
    asyncio.run(migrate())
    sys.exit(0)
