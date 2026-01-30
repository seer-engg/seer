"""Early adopter tracking service."""
from tortoise.transactions import in_transaction

from seer.database.subscription_models import (
    BillingSubscription,
    EarlyAdopterCounter,
    SubscriptionTier,
)
from seer.logger import get_logger

logger = get_logger("api.subscriptions.early_adopter_service")
EARLY_ADOPTER_LIMIT = 50


async def check_and_claim_early_adopter_slot(
    tier: SubscriptionTier, billing_subscription: BillingSubscription
) -> bool:
    """Atomically check if early adopter slots available and claim one. Returns True if claimed."""
    if tier != SubscriptionTier.PRO:
        return False

    # If user was previously early adopter, maintain status
    if billing_subscription.is_early_adopter:
        logger.info("User is returning early adopter, maintaining status")
        return True

    # If user already has Pro subscription, not eligible
    if billing_subscription.tier == SubscriptionTier.PRO:
        logger.info("User already has Pro subscription, not eligible")
        return False

    async with in_transaction() as conn:
        # Lock counter row for atomic update
        counter = await EarlyAdopterCounter.select_for_update().using_db(conn).get(tier=tier.value)

        if counter.count >= EARLY_ADOPTER_LIMIT:
            logger.info("Early adopter limit reached for tier %s", tier.value)
            return False

        counter.count += 1
        await counter.save(update_fields=["count"], using_db=conn)

        logger.info(
            "Claimed early adopter slot %d/%d for tier %s",
            counter.count,
            EARLY_ADOPTER_LIMIT,
            tier.value,
        )
        return True


async def get_early_adopter_count(tier: SubscriptionTier) -> int:
    """Get current early adopter count for a tier."""
    counter = await EarlyAdopterCounter.get_or_none(tier=tier.value)
    return counter.count if counter else 0
