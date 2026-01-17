#!/usr/bin/env python
"""Ensure Stripe products and prices exist for all subscription tiers."""

from seer.api.subscriptions.pricing_catalog import create_prices_in_stripe
from seer.config import config


def main() -> int:
    if not config.stripe_secret_key:
        print("Stripe secret key is not configured. Set STRIPE_SECRET_KEY.")  # noqa: T201
        return 1

    try:
        price_ids = create_prices_in_stripe()
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to ensure Stripe catalog: {exc}")  # noqa: T201
        return 1

    print("Ensured Stripe products and prices exist:")  # noqa: T201
    for lookup_key, price_id in sorted(price_ids.items()):
        print(f"- {lookup_key}: {price_id}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
