"""Subscription pricing catalog and Stripe price helpers."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

import stripe
from pydantic import BaseModel

from seer.config import config
from seer.logger import get_logger

logger = get_logger("api.subscriptions.pricing_catalog")


class PriceInfo(BaseModel):
    """Price information for a billing cycle."""
    price: int
    price_id: Optional[str] = None


class TierPricing(BaseModel):
    """Pricing information for a subscription tier."""
    tier: str
    name: str
    monthly: PriceInfo
    annual: PriceInfo


@dataclass(frozen=True)
class ProductDefinition:
    """Static definition used to construct Stripe products."""
    tier: str
    name: str
    description: Optional[str] = None


@dataclass(frozen=True)
class PriceDefinition:
    """Static definition used to construct Stripe prices."""
    tier: str
    name: str
    interval: str  # month or year
    amount: int
    lookup_key: str
    currency: str = "usd"


PRODUCT_DEFINITIONS: dict[str, ProductDefinition] = {
    "pro": ProductDefinition(
        tier="pro",
        name="Seer Pro",
        description="Pro tier subscription for Seer",
    ),
    "pro_plus": ProductDefinition(
        tier="pro_plus",
        name="Seer Pro+",
        description="Pro+ tier subscription for Seer",
    )
}

PRICE_DEFINITIONS: tuple[PriceDefinition, ...] = (
    PriceDefinition(
        tier="pro",
        name="Pro",
        interval="month",
        amount=20,
        lookup_key="pro_monthly",
    ),
    PriceDefinition(
        tier="pro",
        name="Pro",
        interval="year",
        amount=200,
        lookup_key="pro_annual",
    ),
    PriceDefinition(
        tier="pro_plus",
        name="Pro+",
        interval="month",
        amount=60,
        lookup_key="pro_plus_monthly",
    ),
    PriceDefinition(
        tier="pro_plus",
        name="Pro+",
        interval="year",
        amount=600,
        lookup_key="pro_plus_annual",
    )
)

# Cache Stripe price IDs to avoid hitting the API on every pricing request.
_PRICE_ID_CACHE: dict[str, str] = {}
_PRICE_ID_CACHE_EXPIRES_AT: Optional[datetime] = None
_PRICE_CACHE_TTL = timedelta(days=30)


def _find_product_by_tier(tier: str) -> Optional[dict]:
    """Find an existing Stripe product for the given tier."""
    if not config.stripe_secret_key:
        return None

    stripe.api_key = stripe.api_key or config.stripe_secret_key

    try:
        response = stripe.Product.list(active=True, limit=100)
    except stripe.error.StripeError as exc:
        logger.error("Failed to list Stripe products: %s", exc)
        return None

    data: Iterable[dict] = response.data if hasattr(response, "data") else response.get("data", [])  # type: ignore[index]
    product_definition = PRODUCT_DEFINITIONS.get(tier)
    for product in data:
        metadata = getattr(product, "metadata", None) or product.get("metadata", {})  # type: ignore[attr-defined]
        product_name = getattr(product, "name", None) or product.get("name")  # type: ignore[attr-defined]
        if metadata.get("tier") == tier:
            return product
        if product_definition and product_name and product_name.lower() == product_definition.name.lower():
            return product
    return None


def _get_or_create_product(definition: ProductDefinition) -> dict:
    """Fetch or create the Stripe product for a tier."""
    existing = _find_product_by_tier(definition.tier)
    if existing:
        if not getattr(existing, "active", True):
            try:
                existing = stripe.Product.modify(existing["id"], active=True)
            except stripe.error.StripeError as exc:
                logger.error("Failed to activate Stripe product %s: %s", existing["id"], exc)
                raise
        return existing

    stripe.api_key = stripe.api_key or config.stripe_secret_key
    try:
        return stripe.Product.create(
            name=definition.name,
            description=definition.description,
            metadata={"tier": definition.tier},
        )
    except stripe.error.StripeError as exc:
        logger.error("Failed to create Stripe product for tier %s: %s", definition.tier, exc)
        raise


def _cache_valid(now: datetime) -> bool:
    """Check if the cached Stripe price IDs are still valid."""
    if not _PRICE_ID_CACHE or _PRICE_ID_CACHE_EXPIRES_AT is None:
        return False
    if now >= _PRICE_ID_CACHE_EXPIRES_AT:
        return False
    return True


def _get_cached_stripe_price_ids() -> dict[str, str]:
    """Return cached Stripe price IDs if valid, otherwise fetch and cache them."""
    global _PRICE_ID_CACHE, _PRICE_ID_CACHE_EXPIRES_AT
    now = datetime.now(timezone.utc)
    if _cache_valid(now):
        return dict(_PRICE_ID_CACHE)

    if not config.stripe_secret_key:
        return {}

    stripe.api_key = stripe.api_key or config.stripe_secret_key

    lookup_keys = [definition.lookup_key for definition in PRICE_DEFINITIONS]
    try:
        response = stripe.Price.list(
            lookup_keys=lookup_keys,
            active=True,
            limit=len(lookup_keys),
        )
    except stripe.error.StripeError as exc:
        logger.error("Failed to list Stripe prices: %s", exc)
        return {}

    data: Iterable[dict] = response.data if hasattr(response, "data") else response.get("data", [])  # type: ignore[index]
    fetched: dict[str, str] = {}

    for price in data:
        lookup_key = price.get("lookup_key")
        price_id = price.get("id")
        if lookup_key in lookup_keys and price_id:
            fetched[lookup_key] = price_id

    missing = [key for key in lookup_keys if key not in fetched]
    if missing:
        logger.warning("Missing Stripe prices for lookup keys: %s", ", ".join(missing))

    if fetched:
        _PRICE_ID_CACHE = fetched
        _PRICE_ID_CACHE_EXPIRES_AT = now + _PRICE_CACHE_TTL

    return fetched


def _resolve_price_ids() -> dict[str, Optional[str]]:
    """
    Resolve price IDs using Stripe (cached).

    Returns:
        Mapping from lookup key to Stripe price ID (may be None).
    """
    if not config.stripe_secret_key:
        return {}
    return _get_cached_stripe_price_ids()


def _build_pricing(price_ids: Dict[str, Optional[str]]) -> list[TierPricing]:
    """
    Build tier pricing objects using the static catalog and provided price IDs.

    Args:
        price_ids: Mapping from lookup key to Stripe price ID (may be None).
    """
    tiers: dict[str, dict[str, Any]] = {}
    for definition in PRICE_DEFINITIONS:
        tier_entry = tiers.setdefault(definition.tier, {"name": definition.name})
        price_info = PriceInfo(
            price=definition.amount,
            price_id=price_ids.get(definition.lookup_key),
        )
        if definition.interval == "month":
            tier_entry["monthly"] = price_info
        else:
            tier_entry["annual"] = price_info

    pricing: list[TierPricing] = []
    for tier_key, values in tiers.items():
        pricing.append(TierPricing(
            tier=tier_key,
            name=values["name"],
            monthly=values["monthly"],
            annual=values["annual"],
        ))
    return pricing


def get_pricing_catalog() -> list[TierPricing]:
    """Return pricing details using Stripe price IDs (cached) or config fallbacks."""
    return _build_pricing(_resolve_price_ids())


def _get_existing_price(lookup_key: str, price_id: Optional[str], product_id: Optional[str]) -> Optional[dict]:
    """Fetch an existing Stripe price by ID or lookup key."""
    if not config.stripe_secret_key:
        return None

    stripe.api_key = stripe.api_key or config.stripe_secret_key

    if price_id:
        try:
            return stripe.Price.retrieve(price_id)
        except stripe.error.StripeError as exc:
            logger.warning(
                "Configured price %s could not be retrieved (%s); falling back to lookup_key search",
                price_id,
                exc,
            )

    try:
        response = stripe.Price.list(lookup_keys=[lookup_key], limit=1, expand=["data.product"])
    except stripe.error.StripeError as exc:
        logger.error("Failed to list Stripe prices for %s: %s", lookup_key, exc)
        raise

    data: Iterable[dict] = response.data if hasattr(response, "data") else response.get("data", [])  # type: ignore[index]
    price = next(iter(data), None)
    if price and product_id:
        price_product_id = getattr(price, "product", None) or price.get("product")  # type: ignore[attr-defined]
        if price_product_id and price_product_id != product_id:
            logger.warning(
                "Stripe price %s uses product %s instead of expected %s",
                lookup_key,
                price_product_id,
                product_id,
            )
    return price


def _create_stripe_price(definition: PriceDefinition, product_id: str) -> dict:
    """Create a Stripe price for the given definition."""
    stripe.api_key = stripe.api_key or config.stripe_secret_key
    try:
        return stripe.Price.create(
            currency=definition.currency,
            unit_amount=definition.amount * 100,
            recurring={"interval": definition.interval},
            lookup_key=definition.lookup_key,
            nickname=f"{definition.name} {definition.interval}",
            product=product_id,
            metadata={"tier": definition.tier},
        )
    except stripe.error.StripeError as exc:
        logger.error("Failed to create Stripe price for %s: %s", definition.lookup_key, exc)
        raise


def create_prices_in_stripe() -> dict[str, str]:
    """
    Create or fetch Stripe prices for all tiers and return their IDs.

    Returns:
        Mapping of lookup_key -> Stripe price ID.

    Raises:
        ValueError: if Stripe secret key is not configured.
        stripe.error.StripeError: if Stripe operations fail.
    """
    global _PRICE_ID_CACHE, _PRICE_ID_CACHE_EXPIRES_AT
    if not config.stripe_secret_key:
        raise ValueError("Stripe secret key is not configured")

    stripe.api_key = stripe.api_key or config.stripe_secret_key
    price_ids: dict[str, str] = {}
    product_ids: dict[str, str] = {}

    for definition in PRICE_DEFINITIONS:
        product_definition = PRODUCT_DEFINITIONS.get(definition.tier)
        if not product_definition:
            raise ValueError(f"No product definition for tier {definition.tier}")

        product_id = product_ids.get(definition.tier)
        if not product_id:
            product = _get_or_create_product(product_definition)
            product_id = product["id"]
            product_ids[definition.tier] = product_id

        existing = _get_existing_price(
            lookup_key=definition.lookup_key,
            price_id=None,
            product_id=product_id,
        )

        if existing and not getattr(existing, "active", True):
            existing = stripe.Price.modify(existing["id"], active=True)

        price = existing or _create_stripe_price(definition, product_id)
        price_ids[definition.lookup_key] = price["id"]
        logger.info(
            "Stripe price ready for %s (%s): %s",
            definition.tier,
            definition.interval,
            price["id"],
        )

    _PRICE_ID_CACHE = price_ids
    _PRICE_ID_CACHE_EXPIRES_AT = datetime.now(timezone.utc) + _PRICE_CACHE_TTL

    return price_ids
