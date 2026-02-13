"""Subscription pricing catalog — Stripe is the single source of truth.

Products and prices are fetched from Stripe, parsed from their metadata,
and cached in-memory with a configurable TTL.  No hardcoded prices,
product names, features, or display labels live in this module.

Stripe metadata conventions
---------------------------
**Product metadata** (set on each Stripe Product):
    tier (required)            : "pro" / "pro_plus" / "ultra"
    display_name (required)    : "Pro" / "Pro+"
    features (required)        : JSON array, e.g. '["Unlimited workflows","Priority support"]'
    sort_order                 : "1", "2" — display ordering
    badge (optional)           : e.g. "MOST POPULAR"
    upgrade_benefits (opt.)    : JSON array for PaymentRequiredModal

**Price metadata** (set on each Stripe Price):
    tier (required)            : "pro" — needed for webhook tier resolution
    variant (required)         : "regular"
    trial_period_days (opt.)   : "14"
    original_price_cents (opt.): "3900" — for strikethrough display on discounted prices
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import stripe
from pydantic import BaseModel

from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.subscriptions.pricing_catalog")

# ---------------------------------------------------------------------------
# Public Pydantic models (returned by the API)
# ---------------------------------------------------------------------------


class PriceInfo(BaseModel):
    """Price information for a single billing cycle."""

    price: int
    price_id: Optional[str] = None
    original_price: Optional[int] = None
    trial_period_days: Optional[int] = None
    lookup_key: Optional[str] = None


class TierPricing(BaseModel):
    """Full pricing + display information for one subscription tier."""

    tier: str
    name: str
    monthly: PriceInfo
    annual: PriceInfo
    features: list[str] = []
    badge: Optional[str] = None
    sort_order: int = 0
    description: Optional[str] = None
    upgrade_benefits: list[str] = []


# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_CACHE_TTL = timedelta(minutes=10)


@dataclass
class _CachedProduct:  # pylint: disable=too-many-instance-attributes  # Reason: mirrors Stripe Product metadata structure
    """Parsed product metadata from a Stripe Product."""

    product_id: str
    tier: str
    display_name: str
    features: list[str] = field(default_factory=list)
    sort_order: int = 0
    badge: Optional[str] = None
    description: Optional[str] = None
    upgrade_benefits: list[str] = field(default_factory=list)


@dataclass
class _CachedPrice:  # pylint: disable=too-many-instance-attributes  # Reason: mirrors Stripe Price metadata structure
    """Parsed price metadata from a Stripe Price."""

    price_id: str
    tier: str
    variant: str  # "regular" or "early_adopter"
    interval: str  # "month" or "year"
    amount: int  # unit_amount in cents
    lookup_key: Optional[str] = None
    trial_period_days: Optional[int] = None
    original_price_cents: Optional[int] = None


@dataclass
class _PricingCache:
    """In-memory cache holding parsed Stripe products and prices."""

    products_by_tier: dict[str, _CachedProduct] = field(default_factory=dict)
    prices: list[_CachedPrice] = field(default_factory=list)
    price_id_to_tier: dict[str, str] = field(default_factory=dict)
    lookup_key_to_price_id: dict[str, str] = field(default_factory=dict)
    expires_at: Optional[datetime] = None


_cache = _PricingCache()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_is_valid() -> bool:
    if _cache.expires_at is None:
        return False
    return datetime.now(timezone.utc) < _cache.expires_at


def invalidate_pricing_cache() -> None:
    """Force the next call to re-fetch from Stripe.  Useful for tests / admin."""
    global _cache  # pylint: disable=global-statement  # Reason: intentional caching pattern
    _cache = _PricingCache()


# ---------------------------------------------------------------------------
# Stripe fetching & parsing
# ---------------------------------------------------------------------------


def _parse_json_metadata(raw: Optional[str]) -> list[str]:
    """Safely parse a JSON-encoded list stored in Stripe metadata."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse JSON metadata: %s", raw)
    return []


def _parse_int_metadata(raw: Optional[str]) -> Optional[int]:
    """Safely parse an integer stored in Stripe metadata."""
    if not raw:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _parse_product(product: dict) -> Optional[_CachedProduct]:
    """Parse a Stripe Product dict into _CachedProduct, or None if tier is missing."""
    metadata = product.get("metadata") or {}
    tier = metadata.get("tier")
    if not tier:
        return None
    return _CachedProduct(
        product_id=product["id"],
        tier=tier,
        display_name=metadata.get("display_name", product.get("name", tier)),
        features=_parse_json_metadata(metadata.get("features")),
        sort_order=int(metadata.get("sort_order", 0)),
        badge=metadata.get("badge") or None,
        description=product.get("description"),
        upgrade_benefits=_parse_json_metadata(metadata.get("upgrade_benefits")),
    )


def _parse_price(price: dict) -> Optional[_CachedPrice]:
    """Parse a Stripe Price dict into _CachedPrice, or None if metadata incomplete.

    The ``tier`` metadata field is required.  If ``variant`` is absent,
    we fall back to inferring it from the lookup_key (``*_early_adopter`` →
    ``"early_adopter"``, everything else → ``"regular"``).  This provides
    backward compatibility with prices that predate the metadata convention.
    """
    metadata = price.get("metadata") or {}
    tier = metadata.get("tier")

    # tier can also live on the associated product
    if not tier:
        product = price.get("product")
        if isinstance(product, dict):
            tier = (product.get("metadata") or {}).get("tier")
    if not tier:
        return None

    variant = metadata.get("variant")
    if not variant:
        # Infer variant from lookup_key for backward compatibility
        lookup_key = price.get("lookup_key") or ""
        variant = "early_adopter" if "early_adopter" in lookup_key else "regular"

    recurring = price.get("recurring") or {}
    interval = recurring.get("interval")
    if interval not in ("month", "year"):
        return None

    return _CachedPrice(
        price_id=price["id"],
        tier=tier,
        variant=variant,
        interval=interval,
        amount=price.get("unit_amount") or 0,
        lookup_key=price.get("lookup_key"),
        trial_period_days=_parse_int_metadata(metadata.get("trial_period_days")),
        original_price_cents=_parse_int_metadata(metadata.get("original_price_cents")),
    )


def _fetch_products_from_stripe() -> dict[str, _CachedProduct]:
    """
    Fetch and parse all active products from Stripe.

    Returns:
        Dictionary mapping tier to cached product data
    """
    products_by_tier: dict[str, _CachedProduct] = {}

    try:
        product_response = stripe.Product.list(active=True, limit=100)
        product_data = product_response.data if hasattr(product_response, "data") else product_response.get("data", [])
        for raw_product in product_data:
            product = _parse_product(raw_product)
            if product:
                products_by_tier[product.tier] = product
    except stripe.error.StripeError as exc:
        logger.error("Failed to list Stripe products: %s", exc)

    return products_by_tier


def _fetch_prices_from_stripe() -> tuple[list[_CachedPrice], dict[str, str], dict[str, str]]:
    """
    Fetch and parse all active prices from Stripe.

    Returns:
        Tuple of (prices list, price_id_to_tier mapping, lookup_key_to_price_id mapping)
    """
    prices: list[_CachedPrice] = []
    price_id_to_tier: dict[str, str] = {}
    lookup_key_to_price_id: dict[str, str] = {}

    try:
        price_response = stripe.Price.list(active=True, limit=100, expand=["data.product"])
        price_data = price_response.data if hasattr(price_response, "data") else price_response.get("data", [])
        for raw_price in price_data:
            parsed = _parse_price(raw_price)
            if parsed:
                prices.append(parsed)
                price_id_to_tier[parsed.price_id] = parsed.tier
                if parsed.lookup_key:
                    lookup_key_to_price_id[parsed.lookup_key] = parsed.price_id
    except stripe.error.StripeError as exc:
        logger.error("Failed to list Stripe prices: %s", exc)

    return prices, price_id_to_tier, lookup_key_to_price_id


def _fetch_and_cache_from_stripe() -> None:
    """Fetch all active products and prices from Stripe, parse metadata, build cache."""
    global _cache  # pylint: disable=global-statement  # Reason: intentional caching pattern

    if not config.stripe_secret_key:
        logger.warning("Stripe secret key not configured — pricing cache empty")
        return

    stripe.api_key = stripe.api_key or config.stripe_secret_key

    # Fetch products and prices from Stripe
    products_by_tier = _fetch_products_from_stripe()
    prices, price_id_to_tier, lookup_key_to_price_id = _fetch_prices_from_stripe()

    # Build and store cache
    _cache = _PricingCache(
        products_by_tier=products_by_tier,
        prices=prices,
        price_id_to_tier=price_id_to_tier,
        lookup_key_to_price_id=lookup_key_to_price_id,
        expires_at=datetime.now(timezone.utc) + _CACHE_TTL,
    )

    logger.info(
        "Pricing cache refreshed: %d products, %d prices",
        len(products_by_tier),
        len(prices),
    )


def _ensure_cache() -> None:
    """Populate the cache if expired or empty."""
    if not _cache_is_valid():
        _fetch_and_cache_from_stripe()


# ---------------------------------------------------------------------------
# Public API — pricing catalog
# ---------------------------------------------------------------------------


def get_pricing_catalog() -> list[TierPricing]:
    """Return a sorted list of ``TierPricing`` objects built from cached Stripe data."""
    _ensure_cache()

    variant = "regular"

    result: list[TierPricing] = []
    for tier, product in _cache.products_by_tier.items():
        monthly: Optional[PriceInfo] = None
        annual: Optional[PriceInfo] = None

        # Collect prices for this tier, preferring the requested variant
        tier_prices = [p for p in _cache.prices if p.tier == tier]
        for interval in ("month", "year"):
            # Try requested variant first, fall back to regular
            price = next(
                (p for p in tier_prices if p.interval == interval and p.variant == variant),
                None,
            )
            if price is None and variant != "regular":
                price = next(
                    (p for p in tier_prices if p.interval == interval and p.variant == "regular"),
                    None,
                )

            if price is None:
                continue

            info = PriceInfo(
                price=price.amount,
                price_id=price.price_id,
                original_price=price.original_price_cents,
                trial_period_days=price.trial_period_days,
                lookup_key=price.lookup_key,
            )
            if interval == "month":
                monthly = info
            else:
                annual = info

        if monthly is None or annual is None:
            logger.warning("Tier %s missing monthly or annual price — skipping", tier)
            continue

        result.append(
            TierPricing(
                tier=tier,
                name=product.display_name,
                monthly=monthly,
                annual=annual,
                features=product.features,
                badge=product.badge,
                sort_order=product.sort_order,
                description=product.description,
                upgrade_benefits=product.upgrade_benefits,
            )
        )

    result.sort(key=lambda t: t.sort_order)
    return result


def get_price_id_for_checkout(
    tier: str,
    interval: str,
) -> Optional[str]:
    """Return the Stripe price ID for a tier + interval combination."""
    _ensure_cache()

    match = next(
        (p for p in _cache.prices if p.tier == tier and p.interval == interval and p.variant == "regular"),
        None,
    )

    return match.price_id if match else None


def get_trial_period_days(
    tier: str,
    interval: str,
) -> Optional[int]:
    """Return the trial period in days from cached Stripe price metadata."""
    _ensure_cache()

    match = next(
        (p for p in _cache.prices if p.tier == tier and p.interval == interval and p.variant == "regular"),
        None,
    )
    return match.trial_period_days if match else None


def get_price_id_to_tier_map() -> dict[str, str]:
    """Return a mapping ``{stripe_price_id: tier}`` for webhook tier resolution."""
    _ensure_cache()
    return dict(_cache.price_id_to_tier)
