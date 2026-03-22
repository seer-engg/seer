"""Unit tests for the Stripe pricing catalog."""
from unittest.mock import patch

import pytest

from seer.api.subscriptions.pricing_catalog import (
    _fetch_prices_from_stripe,
    get_price_id_for_checkout,
    get_pricing_catalog,
    invalidate_pricing_cache,
)


@pytest.mark.unit
def test_fetch_prices_skips_active_price_from_inactive_product():
    """Prices from inactive products must never enter the catalog cache."""
    inactive_product_price = {
        "id": "price_inactive_product_month",
        "product": {
            "id": "prod_inactive",
            "active": False,
            "metadata": {"tier": "pro"},
        },
        "metadata": {"tier": "pro", "variant": "regular"},
        "recurring": {"interval": "month", "usage_type": "licensed"},
        "unit_amount": 2900,
        "lookup_key": "pro_monthly_old",
    }
    active_product_price = {
        "id": "price_active_product_month",
        "product": {
            "id": "prod_active",
            "active": True,
            "metadata": {"tier": "pro"},
        },
        "metadata": {"tier": "pro", "variant": "regular"},
        "recurring": {"interval": "month", "usage_type": "licensed"},
        "unit_amount": 3900,
        "lookup_key": "pro_monthly_current",
    }
    mock_response = type("PriceList", (), {"data": [inactive_product_price, active_product_price]})()

    with patch("seer.api.subscriptions.pricing_catalog.stripe.Price.list", return_value=mock_response):
        prices, price_id_to_tier, lookup_key_to_price_id, overage_metered_price = _fetch_prices_from_stripe({"prod_active"})

    assert [price.price_id for price in prices] == ["price_active_product_month"]
    assert price_id_to_tier == {"price_active_product_month": "pro"}
    assert lookup_key_to_price_id == {"pro_monthly_current": "price_active_product_month"}
    assert overage_metered_price is None


@pytest.mark.unit
def test_get_pricing_catalog_uses_prices_from_active_product_only():
    """Catalog selection should not surface a stale price that shares the same tier metadata."""
    invalidate_pricing_cache()

    product_response = type(
        "ProductList",
        (),
        {
            "data": [
                {
                    "id": "prod_active",
                    "active": True,
                    "name": "Pro",
                    "description": "Current product",
                    "metadata": {
                        "tier": "pro",
                        "display_name": "Pro",
                        "features": "[\"Unlimited workflows\"]",
                        "sort_order": "1",
                    },
                }
            ]
        },
    )()
    price_response = type(
        "PriceList",
        (),
        {
            "data": [
                {
                    "id": "price_inactive_month",
                    "product": {
                        "id": "prod_inactive",
                        "active": False,
                        "metadata": {"tier": "pro"},
                    },
                    "metadata": {"tier": "pro", "variant": "regular"},
                    "recurring": {"interval": "month", "usage_type": "licensed"},
                    "unit_amount": 2900,
                    "lookup_key": "pro_monthly_old",
                },
                {
                    "id": "price_active_month",
                    "product": {
                        "id": "prod_active",
                        "active": True,
                        "metadata": {"tier": "pro"},
                    },
                    "metadata": {"tier": "pro", "variant": "regular"},
                    "recurring": {"interval": "month", "usage_type": "licensed"},
                    "unit_amount": 3900,
                    "lookup_key": "pro_monthly_current",
                },
                {
                    "id": "price_active_year",
                    "product": {
                        "id": "prod_active",
                        "active": True,
                        "metadata": {"tier": "pro"},
                    },
                    "metadata": {"tier": "pro", "variant": "regular"},
                    "recurring": {"interval": "year", "usage_type": "licensed"},
                    "unit_amount": 39000,
                    "lookup_key": "pro_yearly_current",
                },
            ]
        },
    )()

    with patch("seer.api.subscriptions.pricing_catalog.config.stripe_secret_key", "sk_test_123"), \
         patch("seer.api.subscriptions.pricing_catalog.stripe.Price.list", return_value=price_response), \
         patch("seer.api.subscriptions.pricing_catalog.stripe.Product.list", return_value=product_response):
        catalog = get_pricing_catalog()
        selected_monthly_price = get_price_id_for_checkout("pro", "month")

    assert len(catalog) == 1
    assert catalog[0].monthly.price_id == "price_active_month"
    assert catalog[0].annual.price_id == "price_active_year"
    assert selected_monthly_price == "price_active_month"
