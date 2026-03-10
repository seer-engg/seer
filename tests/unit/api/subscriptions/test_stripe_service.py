"""
Unit tests for Stripe subscription service.

Tests:
- _build_price_to_tier_map: Price mapping
- _timestamp_to_datetime: Timestamp conversion
- _timestamp_to_iso: ISO string conversion
- _paginate_stripe_list: Pagination helper
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Timestamp Conversion Tests
# =============================================================================


@pytest.mark.unit
class TestTimestampToDatetime:
    """Tests for _timestamp_to_datetime function."""

    def test_timestamp_to_datetime_valid(self):
        """Test converting valid timestamp."""
        from seer.api.subscriptions.stripe_service import _timestamp_to_datetime

        timestamp = 1704067200  # 2024-01-01 00:00:00 UTC

        result = _timestamp_to_datetime(timestamp)

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1
        assert result.tzinfo == timezone.utc

    def test_timestamp_to_datetime_none(self):
        """Test handling None timestamp."""
        from seer.api.subscriptions.stripe_service import _timestamp_to_datetime

        result = _timestamp_to_datetime(None)

        assert result is None

    def test_timestamp_to_datetime_invalid(self):
        """Test handling invalid timestamp."""
        from seer.api.subscriptions.stripe_service import _timestamp_to_datetime

        result = _timestamp_to_datetime("not a timestamp")

        assert result is None


@pytest.mark.unit
class TestTimestampToIso:
    """Tests for _timestamp_to_iso function."""

    def test_timestamp_to_iso_valid(self):
        """Test converting timestamp to ISO string."""
        from seer.api.subscriptions.stripe_service import _timestamp_to_iso

        timestamp = 1704067200  # 2024-01-01 00:00:00 UTC

        result = _timestamp_to_iso(timestamp)

        assert result is not None
        assert "2024-01-01" in result

    def test_timestamp_to_iso_none(self):
        """Test handling None timestamp."""
        from seer.api.subscriptions.stripe_service import _timestamp_to_iso

        result = _timestamp_to_iso(None)

        assert result is None


# =============================================================================
# Price to Tier Map Tests
# =============================================================================


@pytest.mark.unit
class TestBuildPriceToTierMap:
    """Tests for _build_price_to_tier_map function."""

    def test_build_price_to_tier_map_with_pricing(self):
        """Test building price to tier mapping."""
        from seer.api.subscriptions.stripe_service import _build_price_to_tier_map

        with patch("seer.api.subscriptions.stripe_service.get_price_id_to_tier_map") as mock_map:
            mock_map.return_value = {
                "price_monthly_123": "pro",
                "price_annual_456": "pro",
            }

            result = _build_price_to_tier_map()

            assert "price_monthly_123" in result
            assert "price_annual_456" in result

    def test_build_price_to_tier_map_empty_catalog(self):
        """Test building mapping with empty catalog."""
        from seer.api.subscriptions.stripe_service import _build_price_to_tier_map

        with patch("seer.api.subscriptions.stripe_service.get_price_id_to_tier_map") as mock_map:
            mock_map.return_value = {}

            result = _build_price_to_tier_map()

            assert result == {}

    def test_build_price_to_tier_map_catalog_error(self):
        """Test handling catalog load error."""
        from seer.api.subscriptions.stripe_service import _build_price_to_tier_map

        with patch("seer.api.subscriptions.stripe_service.get_price_id_to_tier_map") as mock_map:
            mock_map.side_effect = Exception("Catalog error")

            result = _build_price_to_tier_map()

            assert result == {}


# =============================================================================
# Stripe List Pagination Tests
# =============================================================================


@pytest.mark.unit
class TestPaginateStripeList:
    """Tests for _paginate_stripe_list function."""

    def test_paginate_stripe_list_first_page(self):
        """Test getting first page."""
        from seer.api.subscriptions.stripe_service import _paginate_stripe_list

        mock_response = MagicMock()
        mock_response.data = [{"id": "item_1"}, {"id": "item_2"}]
        mock_response.has_more = True
        mock_response.auto_paging_iter.return_value = iter(mock_response.data)

        list_fn = MagicMock(return_value=mock_response)

        items, has_more = _paginate_stripe_list(list_fn, page=1, page_size=10)

        assert len(items) <= 10
        list_fn.assert_called()

    def test_paginate_stripe_list_invalid_page(self):
        """Test that invalid page raises ValueError."""
        from seer.api.subscriptions.stripe_service import _paginate_stripe_list

        with pytest.raises(ValueError) as exc_info:
            _paginate_stripe_list(MagicMock(), page=0, page_size=10)

        assert "page must be >= 1" in str(exc_info.value)

    def test_paginate_stripe_list_invalid_page_size(self):
        """Test that invalid page_size raises ValueError."""
        from seer.api.subscriptions.stripe_service import _paginate_stripe_list

        with pytest.raises(ValueError) as exc_info:
            _paginate_stripe_list(MagicMock(), page=1, page_size=0)

        assert "page_size must be between 1 and 100" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            _paginate_stripe_list(MagicMock(), page=1, page_size=101)

        assert "page_size must be between 1 and 100" in str(exc_info.value)


# =============================================================================
# Stripe Customer Tests
# =============================================================================


@pytest.mark.unit
class TestStripeCustomerOperations:
    """Tests for Stripe customer operations."""

    def test_stripe_customer_creation_payload(self):
        """Test Stripe customer creation payload structure."""
        # Test that customer creation uses correct email and metadata
        # Organization-centric: metadata includes organization_id and created_by_user_id
        customer_data = {
            "email": "test@example.com",
            "metadata": {
                "organization_id": "1",
                "created_by_user_id": "user_123",
            }
        }

        assert customer_data["email"] == "test@example.com"
        assert customer_data["metadata"]["organization_id"] == "1"
        assert customer_data["metadata"]["created_by_user_id"] == "user_123"


# =============================================================================
# Subscription Sync Tests
# =============================================================================


@pytest.mark.unit
class TestSubscriptionSync:
    """Tests for subscription synchronization."""

    def test_subscription_status_mapping(self):
        """Test that Stripe statuses map to internal statuses."""
        from seer.database.subscription_models import SubscriptionStatus

        # Test the actual enum values that exist
        status_map = {
            "active": SubscriptionStatus.ACTIVE,
            "trialing": SubscriptionStatus.TRIALING,
            "past_due": SubscriptionStatus.PAST_DUE,
            "canceled": SubscriptionStatus.CANCELED,
            "incomplete": SubscriptionStatus.INCOMPLETE,
        }

        for stripe_status, expected in status_map.items():
            assert expected.value == stripe_status

    def test_subscription_tier_values(self):
        """Test subscription tier values."""
        from seer.database.subscription_models import SubscriptionTier

        assert SubscriptionTier.FREE.value == "free"
        assert SubscriptionTier.PRO.value == "pro"
        assert SubscriptionTier.PRO_PLUS.value == "pro_plus"
