#!/usr/bin/env python3
"""
Unified org billing migration script.

Operations:
- transfer: Transfer subscriptions from personal to team orgs
- backfill-payment-method: Set has_payment_method flag for orgs with paid subscriptions
- backfill-usage: Set organization on LLMUsageRecord entries
- sync-counters: Compute org-level UsageCounter aggregates
- all: Run all operations in sequence

Usage:
    uv run scripts/org_billing_migration.py transfer --dry-run
    uv run scripts/org_billing_migration.py backfill-payment-method --dry-run
    uv run scripts/org_billing_migration.py backfill-usage --dry-run --batch-size 500
    uv run scripts/org_billing_migration.py sync-counters --dry-run
    uv run scripts/org_billing_migration.py all --dry-run
"""
import argparse
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

from tortoise import Tortoise
from tortoise.functions import Count, Sum

from seer.database.config import TORTOISE_ORM
from seer.database import User
from seer.database.organization_models import Organization, OrganizationType
from seer.database.subscription_models import (
    BillingSubscription,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.database.usage_models import LLMUsageRecord, ResourceType, UsageCounter
from seer.logger import get_logger
from seer.observability.service import get_billing_period_for_org

logger = get_logger(__name__)


@dataclass
class MigrationStats:
    """Track migration statistics across all operations."""

    # Transfer stats
    users_processed: int = 0
    subscriptions_transferred: int = 0
    users_skipped_no_team: int = 0
    users_skipped_no_paid_sub: int = 0
    users_skipped_already_transferred: int = 0

    # Payment method backfill stats
    payment_method_orgs_updated: int = 0
    payment_method_orgs_skipped: int = 0

    # Usage backfill stats
    usage_records_updated: int = 0
    usage_records_skipped: int = 0

    # Counter sync stats
    orgs_synced: int = 0

    errors: list[str] = field(default_factory=list)


async def init_db():
    """Initialize database connection."""
    await Tortoise.init(config=TORTOISE_ORM)


async def close_db():
    """Close database connection."""
    await Tortoise.close_connections()


# =============================================================================
# Operation 1: Transfer Subscriptions
# =============================================================================


async def get_user_primary_team_org(user: User) -> Organization | None:
    """Get the user's primary (oldest) team organization."""
    return await Organization.filter(
        owner=user,
        type=OrganizationType.TEAM,
    ).order_by("created_at").first()


async def has_transferable_subscription(org: Organization) -> bool:
    """Check if the organization has a paid subscription that can be transferred."""
    # Must have a Stripe customer to transfer
    if not org.stripe_customer_id:
        return False

    subscription = await BillingSubscription.get_or_none(organization=org)
    if not subscription:
        return False

    return (
        subscription.tier != SubscriptionTier.FREE
        and subscription.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING]
    )


async def team_needs_subscription(org: Organization) -> bool:
    """Check if the team org needs a subscription (has FREE or no subscription)."""
    subscription = await BillingSubscription.get_or_none(organization=org)
    if not subscription:
        return True
    return subscription.tier == SubscriptionTier.FREE


async def transfer_subscriptions(
    dry_run: bool,
    user_id: str | None,
    stats: MigrationStats,
) -> None:
    """
    Find and transfer subscriptions from personal orgs to team orgs.

    For each user who owns a team org:
    - Check if their personal org has a paid subscription
    - Check if their team org needs a subscription
    - Transfer the subscription using transfer_subscription_between_orgs
    """
    # Import here to avoid circular imports at module load time
    from seer.api.subscriptions.stripe_service import transfer_subscription_between_orgs

    logger.info("=" * 60)
    logger.info("OPERATION: Transfer Subscriptions")
    logger.info("=" * 60)

    # Build user query
    if user_id:
        users = await User.filter(user_id=user_id).all()
        logger.info("Filtering to user: %s", user_id)
    else:
        # Get users who own at least one team org
        team_org_owner_ids = await Organization.filter(
            type=OrganizationType.TEAM,
        ).values_list("owner_id", flat=True)
        unique_owner_ids = list(set(team_org_owner_ids))

        if not unique_owner_ids:
            logger.info("No users with team organizations found.")
            return

        users = await User.filter(id__in=unique_owner_ids).all()

    logger.info("Found %d users to process", len(users))

    for user in users:
        logger.info("Processing user: %s (id=%d)", user.user_id, user.id)
        stats.users_processed += 1

        # Step 1: Check if user has a team org
        team_org = await get_user_primary_team_org(user)
        if not team_org:
            logger.info("  Skipped: No team organization owned by this user")
            stats.users_skipped_no_team += 1
            continue

        logger.info("  Found team org: %s (id=%d)", team_org.name, team_org.id)

        # Step 2: Check if team already has a paid subscription
        if not await team_needs_subscription(team_org):
            logger.info("  Skipped: Team org already has a paid subscription")
            stats.users_skipped_already_transferred += 1
            continue

        # Step 3: Check if personal org has a transferable subscription
        personal_org = await Organization.get_or_none(
            owner=user,
            type=OrganizationType.PERSONAL,
        )

        if not personal_org:
            logger.info("  Skipped: No personal organization found")
            stats.users_skipped_no_paid_sub += 1
            continue

        if not await has_transferable_subscription(personal_org):
            logger.info("  Skipped: Personal org has no transferable subscription (no Stripe customer or no paid sub)")
            stats.users_skipped_no_paid_sub += 1
            continue

        # Step 4: Transfer the subscription
        if dry_run:
            logger.info("  [DRY RUN] Would transfer subscription from org %d to org %d",
                       personal_org.id, team_org.id)
        else:
            try:
                await transfer_subscription_between_orgs(personal_org, team_org)
                logger.info("  Transferred subscription from org %d to org %d",
                           personal_org.id, team_org.id)
            except Exception as e:
                error_msg = f"Failed to transfer for user {user.user_id}: {e}"
                logger.error("  ERROR: %s", error_msg)
                stats.errors.append(error_msg)
                continue

        stats.subscriptions_transferred += 1

    logger.info("")
    logger.info("Transfer complete: %d subscriptions transferred", stats.subscriptions_transferred)


# =============================================================================
# Operation 2: Backfill Payment Method Flags
# =============================================================================


async def backfill_payment_method_flags(
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """
    Set has_payment_method=True for organizations with active paid subscriptions.

    This fixes the payment method gate issue where users with valid paid subscriptions
    are blocked because has_payment_method was not set during migration.
    """
    logger.info("=" * 60)
    logger.info("OPERATION: Backfill Payment Method Flags")
    logger.info("=" * 60)

    # Find all orgs with active paid subscriptions that don't have payment method flag set
    paid_subs = await BillingSubscription.filter(
        tier__not=SubscriptionTier.FREE,
        status__in=[SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING],
        stripe_subscription_id__isnull=False,
    ).prefetch_related("organization")

    logger.info("Found %d active paid subscriptions", len(paid_subs))

    for sub in paid_subs:
        org = sub.organization

        if org.has_payment_method:
            logger.info("  Org %d (%s): already has has_payment_method=True, skipping",
                       org.id, org.name)
            stats.payment_method_orgs_skipped += 1
            continue

        if dry_run:
            logger.info("  [DRY RUN] Would set has_payment_method=True for org %d (%s)",
                       org.id, org.name)
        else:
            org.has_payment_method = True
            org.payment_method_added_at = datetime.now(timezone.utc)
            await org.save()
            logger.info("  Set has_payment_method=True for org %d (%s)", org.id, org.name)

        stats.payment_method_orgs_updated += 1

    logger.info("")
    logger.info("Payment method backfill complete: %d orgs updated, %d skipped",
               stats.payment_method_orgs_updated, stats.payment_method_orgs_skipped)


# =============================================================================
# Operation 3: Backfill Usage Organization (formerly Operation 2)
# =============================================================================


async def backfill_usage_org(
    dry_run: bool,
    batch_size: int,
    stats: MigrationStats,
) -> None:
    """
    Set organization on existing LLMUsageRecord entries that don't have it.

    For records without organization, sets to user's personal org.
    This is conservative - we don't know which org context was used originally.
    """
    logger.info("=" * 60)
    logger.info("OPERATION: Backfill LLMUsageRecord.organization")
    logger.info("=" * 60)

    # Count records needing backfill
    total_count = await LLMUsageRecord.filter(organization_id__isnull=True).count()
    logger.info("Found %d LLMUsageRecord entries without organization", total_count)

    if total_count == 0:
        logger.info("No records to backfill")
        return

    if dry_run:
        logger.info("[DRY RUN] Would update %d records", total_count)
        stats.usage_records_updated = total_count
        return

    offset = 0

    while offset < total_count:
        # Fetch batch of records
        records = await LLMUsageRecord.filter(
            organization_id__isnull=True
        ).offset(offset).limit(batch_size).prefetch_related("user")

        if not records:
            break

        for record in records:
            if record.user is None:
                logger.warning("Skipping record %d: no user", record.id)
                stats.usage_records_skipped += 1
                continue

            # Get user's personal org
            personal_org = await Organization.get_or_none(
                owner=record.user,
                type=OrganizationType.PERSONAL,
            )

            if personal_org:
                record.organization = personal_org
                await record.save()
                stats.usage_records_updated += 1
            else:
                logger.warning(
                    "No personal org found for user %d (record %d)",
                    record.user.id,
                    record.id,
                )
                stats.usage_records_skipped += 1

        offset += batch_size
        logger.info("Processed %d/%d records...", min(offset, total_count), total_count)

    logger.info("")
    logger.info("Backfill complete: %d records updated, %d skipped",
               stats.usage_records_updated, stats.usage_records_skipped)


# =============================================================================
# Operation 4: Sync Org Counters (formerly Operation 3)
# =============================================================================


async def sync_org_counters(
    dry_run: bool,
    stats: MigrationStats,
) -> None:
    """
    Compute org-level counters from existing LLMUsageRecord data.

    For each team organization, aggregates LLM usage for the current billing
    period and creates/updates org-level UsageCounter entries.
    """
    logger.info("=" * 60)
    logger.info("OPERATION: Sync Org-Level UsageCounters")
    logger.info("=" * 60)

    # Get all team organizations
    team_orgs = await Organization.filter(type=OrganizationType.TEAM)
    logger.info("Found %d team organizations", len(team_orgs))

    if not team_orgs:
        logger.info("No team organizations to sync")
        return

    for org in team_orgs:
        # Get org's billing period
        period_start, period_end = await get_billing_period_for_org(org)

        # Aggregate LLM usage for this org in this period
        result = await LLMUsageRecord.filter(
            organization=org,
            created_at__gte=period_start,
            created_at__lt=period_end,
        ).annotate(
            total_cost=Sum("cost"),
            call_count=Count("id"),
        ).values("total_cost", "call_count")

        if result and result[0]["total_cost"]:
            total_cost = Decimal(str(result[0]["total_cost"]))
            call_count = result[0]["call_count"]
        else:
            total_cost = Decimal("0.0")
            call_count = 0

        logger.info(
            "Org %d (%s): %d calls, $%.4f in period %s to %s",
            org.id,
            org.name,
            call_count,
            total_cost,
            period_start.isoformat(),
            period_end.isoformat(),
        )

        if dry_run:
            logger.info("[DRY RUN] Would update/create counter for org %d", org.id)
            stats.orgs_synced += 1
            continue

        # Create or update org-level counter
        counter, created = await UsageCounter.get_or_create(
            user=None,  # Org-level aggregate
            organization=org,
            resource_type=ResourceType.LLM_CREDITS,
            period_start=period_start,
            period_end=period_end,
            defaults={"count": call_count, "value": total_cost},
        )

        if not created:
            # Update existing counter
            await UsageCounter.filter(id=counter.id).update(
                count=call_count,
                value=total_cost,
            )
            logger.info("Updated existing counter %d for org %d", counter.id, org.id)
        else:
            logger.info("Created new counter %d for org %d", counter.id, org.id)

        stats.orgs_synced += 1

    logger.info("")
    logger.info("Counter sync complete: %d organizations processed", stats.orgs_synced)


# =============================================================================
# Main Entry Point
# =============================================================================


def print_summary(stats: MigrationStats, dry_run: bool) -> None:
    """Print migration summary."""
    mode = "[DRY RUN] " if dry_run else ""

    print("")
    print("=" * 60)
    print(f"{mode}MIGRATION SUMMARY")
    print("=" * 60)

    print("\nSubscription Transfer:")
    print(f"  Users processed:              {stats.users_processed}")
    print(f"  Subscriptions transferred:    {stats.subscriptions_transferred}")
    print(f"  Skipped (no team org):        {stats.users_skipped_no_team}")
    print(f"  Skipped (no paid sub):        {stats.users_skipped_no_paid_sub}")
    print(f"  Skipped (already transferred): {stats.users_skipped_already_transferred}")

    print("\nPayment Method Backfill:")
    print(f"  Orgs updated:                 {stats.payment_method_orgs_updated}")
    print(f"  Orgs skipped (already set):   {stats.payment_method_orgs_skipped}")

    print("\nUsage Backfill:")
    print(f"  Records updated:              {stats.usage_records_updated}")
    print(f"  Records skipped:              {stats.usage_records_skipped}")

    print("\nCounter Sync:")
    print(f"  Organizations synced:         {stats.orgs_synced}")

    if stats.errors:
        print(f"\nERRORS ({len(stats.errors)}):")
        for error in stats.errors:
            print(f"  - {error}")

    print("=" * 60)

    if dry_run:
        print("\n(No changes were written - re-run without --dry-run to apply)")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified org billing migration script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Operations:
  transfer                Transfer subscriptions from personal orgs to team orgs
  backfill-payment-method Set has_payment_method flag for orgs with paid subscriptions
  backfill-usage          Set organization on LLMUsageRecord entries without it
  sync-counters           Compute org-level UsageCounter aggregates
  all                     Run all operations in sequence

Examples:
  uv run scripts/org_billing_migration.py transfer --dry-run
  uv run scripts/org_billing_migration.py backfill-payment-method --dry-run
  uv run scripts/org_billing_migration.py backfill-usage --batch-size 500
  uv run scripts/org_billing_migration.py all --dry-run
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Operation to perform")

    # Transfer subcommand
    transfer_parser = subparsers.add_parser(
        "transfer",
        help="Transfer subscriptions from personal orgs to team orgs",
    )
    transfer_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )
    transfer_parser.add_argument(
        "--user-id",
        type=str,
        help="Only process a specific user by Clerk user_id",
    )

    # Backfill payment method subcommand
    payment_method_parser = subparsers.add_parser(
        "backfill-payment-method",
        help="Set has_payment_method flag for orgs with paid subscriptions",
    )
    payment_method_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )

    # Backfill usage subcommand
    backfill_parser = subparsers.add_parser(
        "backfill-usage",
        help="Set organization on LLMUsageRecord entries",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )
    backfill_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of records to process per batch (default: 1000)",
    )

    # Sync counters subcommand
    sync_parser = subparsers.add_parser(
        "sync-counters",
        help="Compute org-level UsageCounter aggregates",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )

    # All subcommand
    all_parser = subparsers.add_parser(
        "all",
        help="Run all operations in sequence",
    )
    all_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )
    all_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for usage backfill (default: 1000)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    dry_run = getattr(args, "dry_run", False)
    mode_label = "[DRY RUN] " if dry_run else ""

    print("=" * 60)
    print(f"{mode_label}ORG BILLING MIGRATION")
    print("=" * 60)
    print(f"Command: {args.command}")
    print("")

    stats = MigrationStats()

    await init_db()

    try:
        if args.command == "transfer":
            await transfer_subscriptions(
                dry_run=dry_run,
                user_id=getattr(args, "user_id", None),
                stats=stats,
            )

        elif args.command == "backfill-payment-method":
            await backfill_payment_method_flags(
                dry_run=dry_run,
                stats=stats,
            )

        elif args.command == "backfill-usage":
            await backfill_usage_org(
                dry_run=dry_run,
                batch_size=getattr(args, "batch_size", 1000),
                stats=stats,
            )

        elif args.command == "sync-counters":
            await sync_org_counters(
                dry_run=dry_run,
                stats=stats,
            )

        elif args.command == "all":
            await transfer_subscriptions(
                dry_run=dry_run,
                user_id=None,
                stats=stats,
            )
            print("")
            await backfill_payment_method_flags(
                dry_run=dry_run,
                stats=stats,
            )
            print("")
            await backfill_usage_org(
                dry_run=dry_run,
                batch_size=getattr(args, "batch_size", 1000),
                stats=stats,
            )
            print("")
            await sync_org_counters(
                dry_run=dry_run,
                stats=stats,
            )

    finally:
        await close_db()

    print_summary(stats, dry_run)


if __name__ == "__main__":
    asyncio.run(main())
