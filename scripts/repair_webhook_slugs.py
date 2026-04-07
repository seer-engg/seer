"""
One-time repair script: regenerate webhook_slug for any trigger_subscription
where trigger_key starts with 'webhook.' but webhook_slug is NULL.

Run with:
    uv run python scripts/repair_webhook_slugs.py

Safe to run multiple times — only affects rows with NULL webhook_slug.
"""

import asyncio
import secrets

from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM
from seer.database import TriggerSubscription


async def main() -> None:
    await Tortoise.init(config=TORTOISE_ORM)

    broken = await TriggerSubscription.filter(
        trigger_key__startswith="webhook.",
        webhook_slug__isnull=True,
    ).all()

    if not broken:
        print("No subscriptions with NULL webhook_slug found — nothing to do.")
        await Tortoise.close_connections()
        return

    print(f"Found {len(broken)} subscription(s) with NULL webhook_slug:")
    for sub in broken:
        slug = secrets.token_urlsafe(32)
        sub.webhook_slug = slug
        await sub.save(update_fields=["webhook_slug"])
        print(f"  Fixed id={sub.id}  workflow={sub.workflow_id}  trigger_key={sub.trigger_key}  slug={slug}")

    await Tortoise.close_connections()
    print("Done.")


asyncio.run(main())
