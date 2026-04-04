"""
One-off script to re-sync Pro+ subscriptions from Stripe.

Populates current_period_start/end and confirms tier resolves correctly
after the pro_plus → pro migration fix.

Usage:
    uv run python scripts/resync_pro_subscriptions.py
"""
from __future__ import annotations

import asyncio
import sys

from tortoise import Tortoise

from seer.config import config
from seer.database.config import TORTOISE_ORM


SUBSCRIPTION_IDS = [
    "sub_1T6C41HF67yAMDTED2kbetzV",  # Billy Mayberry (sales@marionbillboards.com)
    "sub_1SyJU1HF67yAMDTEiXRR05D1",  # Frederica Carrier (sneacfulgiftsworkflow@gmail.com)
]

# Sub ID 6 (Lokesh) has no stripe_subscription_id — early adopter, no Stripe sync needed.


async def main() -> None:
    # Lazy import to avoid circular deps
    from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe

    await Tortoise.init(config=TORTOISE_ORM)

    print(f"Re-syncing {len(SUBSCRIPTION_IDS)} subscriptions from Stripe...\n")

    for sub_id in SUBSCRIPTION_IDS:
        print(f"  Syncing {sub_id} ...")
        result = await sync_subscription_from_stripe(sub_id)
        if result:
            print(
                f"    ✓ org={result.organization_id}  tier={result.tier}  "
                f"status={result.status}  "
                f"period_start={result.current_period_start}  "
                f"period_end={result.current_period_end}"
            )
        else:
            print(f"    ✗ Failed — check logs for details")

    await Tortoise.close_connections()
    print("\nDone.")


if __name__ == "__main__":
    if not config.stripe_secret_key:
        print("ERROR: STRIPE_SECRET_KEY not set. Cannot fetch from Stripe.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main())
