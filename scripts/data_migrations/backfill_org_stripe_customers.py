#!/usr/bin/env python3
"""
Backfill missing organization Stripe customer links for legacy paid subscriptions.

This script targets paid personal/team organizations that still have an active or
trialing subscription but are missing the org-centric `organizations.stripe_customer_id`
 linkage introduced in the V2 billing migration.

It resolves Stripe customer ids from durable webhook history, creates any missing
`stripe_customers` audit rows, and links the organization to that audit record.

Usage:
    uv run python scripts/data_migrations/backfill_org_stripe_customers.py --dry-run

    uv run python scripts/data_migrations/backfill_org_stripe_customers.py

    uv run python scripts/data_migrations/backfill_org_stripe_customers.py --organization-id 21
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field

from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM
from seer.database.organization_models import Organization
from seer.database.subscription_models import (
    BillingSubscription,
    StripeCustomer,
    StripeWebhookEvent,
    SubscriptionStatus,
    SubscriptionTier,
)
from seer.logger import get_logger

logger = get_logger(__name__)

RELEVANT_WEBHOOK_TYPES = {
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.trial_will_end",
    "invoice.created",
    "invoice.finalized",
    "invoice.paid",
    "invoice.payment_succeeded",
    "invoice.updated",
}


@dataclass
class BackfillStats:
    """Track migration outcomes."""

    scanned: int = 0
    linked_existing_customer: int = 0
    created_customer_rows: int = 0
    linked_organizations: int = 0
    skipped_missing_subscription_id: int = 0
    skipped_missing_customer: int = 0
    skipped_conflict: int = 0
    errors: list[str] = field(default_factory=list)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the backfill without writing any changes.",
    )
    parser.add_argument(
        "--organization-id",
        type=int,
        help="Restrict the backfill to a single organization id.",
    )
    return parser.parse_args()


async def init_db() -> None:
    """Initialize database connection."""
    await Tortoise.init(config=TORTOISE_ORM)


async def close_db() -> None:
    """Close database connections."""
    await Tortoise.close_connections()


def _extract_customer_id(event: StripeWebhookEvent) -> tuple[str | None, str | None]:
    """Extract subscription id and customer id from a webhook event payload."""
    payload = event.payload or {}
    data = payload.get("data") or {}
    obj = data.get("object") or {}

    customer_id = obj.get("customer")
    if not isinstance(customer_id, str) or not customer_id:
        return None, None

    object_id = obj.get("id")
    subscription_id = obj.get("subscription")

    if isinstance(object_id, str) and object_id.startswith("sub_"):
        return object_id, customer_id

    if isinstance(subscription_id, str) and subscription_id:
        return subscription_id, customer_id

    return None, None


async def _build_subscription_customer_map() -> dict[str, str]:
    """Build a latest-known mapping of Stripe subscription id -> Stripe customer id."""
    mapping: dict[str, str] = {}

    events = await StripeWebhookEvent.filter(
        type__in=list(RELEVANT_WEBHOOK_TYPES),
    ).order_by("-created_at")

    for event in events:
        subscription_id, customer_id = _extract_customer_id(event)
        if subscription_id and customer_id and subscription_id not in mapping:
            mapping[subscription_id] = customer_id

    return mapping


async def _find_conflicting_org(stripe_customer_row: StripeCustomer, org_id: int) -> Organization | None:
    """Return an already-linked organization if the Stripe customer is owned elsewhere."""
    return await Organization.filter(stripe_customer_id=stripe_customer_row.id).exclude(id=org_id).first()


async def backfill_missing_org_stripe_customers(
    *,
    dry_run: bool,
    organization_id: int | None,
) -> BackfillStats:
    """Backfill missing org Stripe customer links using webhook history."""
    stats = BackfillStats()
    subscription_customer_map = await _build_subscription_customer_map()

    candidate_query = BillingSubscription.filter(
        tier__in=[SubscriptionTier.PRO, SubscriptionTier.PRO_PLUS],
        status__in=[SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING],
    ).prefetch_related("organization__owner")

    if organization_id is not None:
        candidate_query = candidate_query.filter(organization_id=organization_id)

    subscriptions = await candidate_query

    for subscription in subscriptions:
        organization = subscription.organization
        if organization is None or organization.stripe_customer_id is not None:
            continue

        stats.scanned += 1

        if not subscription.stripe_subscription_id:
            logger.warning(
                "Org %s (%s) has paid subscription %s but no stripe_subscription_id; skipping",
                organization.id,
                organization.name,
                subscription.id,
            )
            stats.skipped_missing_subscription_id += 1
            continue

        customer_id = subscription_customer_map.get(subscription.stripe_subscription_id)
        if not customer_id:
            logger.warning(
                "Org %s (%s) subscription %s has no customer id in webhook history; skipping",
                organization.id,
                organization.name,
                subscription.stripe_subscription_id,
            )
            stats.skipped_missing_customer += 1
            continue

        stripe_customer = await StripeCustomer.get_or_none(stripe_customer_id=customer_id)
        created_customer = False
        if stripe_customer is None:
            created_customer = True
            stripe_customer = StripeCustomer(
                stripe_customer_id=customer_id,
                created_by_user=organization.owner,
            )

        conflicting_org = None
        if not created_customer:
            conflicting_org = await _find_conflicting_org(stripe_customer, organization.id)

        if conflicting_org is not None:
            logger.warning(
                "Stripe customer %s already linked to org %s; skipping org %s",
                customer_id,
                conflicting_org.id,
                organization.id,
            )
            stats.skipped_conflict += 1
            continue

        if dry_run:
            logger.info(
                "[DRY RUN] Would link org %s (%s) to Stripe customer %s",
                organization.id,
                organization.name,
                customer_id,
            )
            if created_customer:
                stats.created_customer_rows += 1
            else:
                stats.linked_existing_customer += 1
            stats.linked_organizations += 1
            continue

        try:
            if created_customer:
                await stripe_customer.save()
                stats.created_customer_rows += 1
            else:
                stats.linked_existing_customer += 1

            organization.stripe_customer = stripe_customer
            await organization.save(update_fields=["stripe_customer_id"])
            stats.linked_organizations += 1

            logger.info(
                "Linked org %s (%s) to Stripe customer %s",
                organization.id,
                organization.name,
                customer_id,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"Failed to backfill org {organization.id}: {exc}"
            logger.error(error)
            stats.errors.append(error)

    return stats


def log_summary(stats: BackfillStats, *, dry_run: bool) -> None:
    """Log a concise summary."""
    mode = "[DRY RUN] " if dry_run else ""
    logger.info("%sStripe customer backfill summary:", mode)
    logger.info("  Candidate orgs scanned: %s", stats.scanned)
    logger.info("  Organizations linked: %s", stats.linked_organizations)
    logger.info("  Existing StripeCustomer rows reused: %s", stats.linked_existing_customer)
    logger.info("  StripeCustomer rows created: %s", stats.created_customer_rows)
    logger.info("  Skipped (missing stripe_subscription_id): %s", stats.skipped_missing_subscription_id)
    logger.info("  Skipped (no customer in webhook history): %s", stats.skipped_missing_customer)
    logger.info("  Skipped (customer already linked elsewhere): %s", stats.skipped_conflict)
    if stats.errors:
        logger.error("  Errors: %s", len(stats.errors))
        for error in stats.errors:
            logger.error("    %s", error)


async def main() -> None:
    """Run the backfill."""
    args = _parse_args()
    await init_db()
    try:
        stats = await backfill_missing_org_stripe_customers(
            dry_run=args.dry_run,
            organization_id=args.organization_id,
        )
        log_summary(stats, dry_run=args.dry_run)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
