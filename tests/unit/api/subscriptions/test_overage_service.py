# pylint: disable=import-outside-toplevel
# Reason: Test file with lazy imports
"""
Unit tests for overage service layer.

Tests cover:
- Overage eligibility checks
- Overage settings management
- Stripe subscription item attachment/detachment
- Usage reporting to Stripe (Billing Meter Events API)
- Spending cap management
- Period reset functionality
"""
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_billing_profile():
    """Create mock billing profile."""
    from seer.database.subscription_models import BillingProfile, BillingProfileType

    profile = MagicMock(spec=BillingProfile)
    profile.id = 1
    profile.profile_type = BillingProfileType.INDIVIDUAL
    profile.stripe_customer_id = "cus_test123"
    profile.has_payment_method = True
    return profile


@pytest.fixture
def mock_subscription():
    """Create mock billing subscription."""
    from seer.database.subscription_models import (
        BillingSubscription,
        SubscriptionStatus,
        SubscriptionTier,
    )

    subscription = MagicMock(spec=BillingSubscription)
    subscription.id = 1
    subscription.tier = SubscriptionTier.PRO
    subscription.status = SubscriptionStatus.ACTIVE
    subscription.stripe_subscription_id = "sub_test123"
    subscription.current_period_start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    return subscription


@pytest.fixture
def mock_overage_settings():
    """Create mock overage settings."""
    from seer.database.overage_models import OverageSettings

    settings = MagicMock(spec=OverageSettings)
    settings.id = 1
    settings.enabled = False
    settings.spending_cap_cents = 5000
    settings.margin_multiplier = Decimal("1.30")
    settings.current_period_overage_cents = 0
    settings.current_period_start = None
    settings.stripe_metered_subscription_item_id = None
    settings.save = AsyncMock()
    return settings


# =============================================================================
# Overage Price ID Tests
# =============================================================================


class TestGetOveragePriceId:
    """Tests for _get_overage_price_id function."""

    def test_returns_price_id_when_available(self):
        """Test returns price ID from pricing catalog."""
        from seer.api.subscriptions.overage_service import _get_overage_price_id

        with patch(
            "seer.api.subscriptions.overage_service.get_overage_metered_price_id"
        ) as mock_get_price:
            mock_get_price.return_value = "price_overage_123"

            result = _get_overage_price_id()

            assert result == "price_overage_123"

    def test_raises_when_no_price_configured(self):
        """Test raises ValueError when no overage price found."""
        from seer.api.subscriptions.overage_service import _get_overage_price_id

        with patch(
            "seer.api.subscriptions.overage_service.get_overage_metered_price_id"
        ) as mock_get_price:
            mock_get_price.return_value = None

            with pytest.raises(ValueError) as exc_info:
                _get_overage_price_id()

            assert "No overage metered price found" in str(exc_info.value)


# =============================================================================
# Overage Eligibility Tests
# =============================================================================


class TestIsOverageEligible:
    """Tests for is_overage_eligible function."""

    @pytest.mark.asyncio
    async def test_eligible_pro_tier_active_with_payment(
        self, mock_subscription, mock_billing_profile
    ):
        """Test PRO tier with active subscription and payment method is eligible."""
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is True

    @pytest.mark.asyncio
    async def test_not_eligible_free_tier(self, mock_subscription, mock_billing_profile):
        """Test FREE tier is not eligible."""
        from seer.database.subscription_models import SubscriptionTier
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_subscription.tier = SubscriptionTier.FREE
        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is False

    @pytest.mark.asyncio
    async def test_not_eligible_canceled_subscription(
        self, mock_subscription, mock_billing_profile
    ):
        """Test canceled subscription is not eligible."""
        from seer.database.subscription_models import SubscriptionStatus
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_subscription.status = SubscriptionStatus.CANCELED
        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is False

    @pytest.mark.asyncio
    async def test_not_eligible_no_payment_method(
        self, mock_subscription, mock_billing_profile
    ):
        """Test not eligible without payment method."""
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_billing_profile.has_payment_method = False
        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is False

    @pytest.mark.asyncio
    async def test_eligible_trialing_status(self, mock_subscription, mock_billing_profile):
        """Test trialing subscription is eligible."""
        from seer.database.subscription_models import SubscriptionStatus
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_subscription.status = SubscriptionStatus.TRIALING
        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is True

    @pytest.mark.asyncio
    async def test_eligible_pro_plus_tier(self, mock_subscription, mock_billing_profile):
        """Test PRO_PLUS tier is eligible."""
        from seer.database.subscription_models import SubscriptionTier
        from seer.api.subscriptions.overage_service import is_overage_eligible

        mock_subscription.tier = SubscriptionTier.PRO_PLUS
        mock_subscription.billing_profile = mock_billing_profile
        mock_subscription.fetch_related = AsyncMock()

        result = await is_overage_eligible(mock_subscription)

        assert result is True


# =============================================================================
# Get Or Create Overage Settings Tests
# =============================================================================


class TestGetOrCreateOverageSettings:
    """Tests for get_or_create_overage_settings function."""

    @pytest.mark.asyncio
    async def test_creates_settings_with_defaults(self, mock_billing_profile):
        """Test creates settings with default values."""
        from seer.api.subscriptions.overage_service import get_or_create_overage_settings

        mock_settings = MagicMock()

        with patch(
            "seer.api.subscriptions.overage_service.OverageSettings"
        ) as mock_overage_cls:
            mock_overage_cls.get_or_create = AsyncMock(return_value=(mock_settings, True))

            result = await get_or_create_overage_settings(mock_billing_profile)

            assert result == mock_settings
            mock_overage_cls.get_or_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_existing_settings(self, mock_billing_profile):
        """Test returns existing settings if already exists."""
        from seer.api.subscriptions.overage_service import get_or_create_overage_settings

        existing_settings = MagicMock()

        with patch(
            "seer.api.subscriptions.overage_service.OverageSettings"
        ) as mock_overage_cls:
            mock_overage_cls.get_or_create = AsyncMock(
                return_value=(existing_settings, False)
            )

            result = await get_or_create_overage_settings(mock_billing_profile)

            assert result == existing_settings


# =============================================================================
# Attach Overage Pricing Tests
# =============================================================================


class TestAttachOveragePricing:
    """Tests for attach_overage_pricing function."""

    @pytest.mark.asyncio
    async def test_attaches_metered_price_to_subscription(self, mock_subscription):
        """Test attaches metered price as subscription item."""
        from seer.api.subscriptions.overage_service import attach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("seer.api.subscriptions.overage_service._get_overage_price_id") as mock_price, \
             patch("stripe.SubscriptionItem.create") as mock_create:

            mock_config.stripe_secret_key = "sk_test_123"
            mock_price.return_value = "price_overage_123"
            mock_create.return_value = MagicMock(id="si_123")

            result = await attach_overage_pricing(mock_subscription)

            assert result == "si_123"
            mock_create.assert_called_once_with(
                subscription="sub_test123",
                price="price_overage_123",
                metadata={"purpose": "llm_overage"},
            )

    @pytest.mark.asyncio
    async def test_returns_none_when_stripe_not_configured(self, mock_subscription):
        """Test returns None when Stripe is not configured."""
        from seer.api.subscriptions.overage_service import attach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config:
            mock_config.stripe_secret_key = None

            result = await attach_overage_pricing(mock_subscription)

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_stripe_subscription_id(self, mock_subscription):
        """Test returns None when subscription has no Stripe ID."""
        from seer.api.subscriptions.overage_service import attach_overage_pricing

        mock_subscription.stripe_subscription_id = None

        with patch("seer.api.subscriptions.overage_service.config") as mock_config:
            mock_config.stripe_secret_key = "sk_test_123"

            result = await attach_overage_pricing(mock_subscription)

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_stripe_error(self, mock_subscription):
        """Test returns None on Stripe API error."""
        import stripe
        from seer.api.subscriptions.overage_service import attach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("seer.api.subscriptions.overage_service._get_overage_price_id") as mock_price, \
             patch("stripe.SubscriptionItem.create") as mock_create:

            mock_config.stripe_secret_key = "sk_test_123"
            mock_price.return_value = "price_overage_123"
            mock_create.side_effect = stripe.error.StripeError("API Error")  # type: ignore[attr-defined]

            result = await attach_overage_pricing(mock_subscription)

            assert result is None


# =============================================================================
# Detach Overage Pricing Tests
# =============================================================================


class TestDetachOveragePricing:
    """Tests for detach_overage_pricing function."""

    @pytest.mark.asyncio
    async def test_detaches_subscription_item(self, mock_subscription):
        """Test detaches metered subscription item."""
        from seer.api.subscriptions.overage_service import detach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("stripe.SubscriptionItem.delete") as mock_delete:

            mock_config.stripe_secret_key = "sk_test_123"

            result = await detach_overage_pricing(mock_subscription, "si_123")

            assert result is True
            mock_delete.assert_called_once_with(
                "si_123",
                proration_behavior="none",
                clear_usage=False,
            )

    @pytest.mark.asyncio
    async def test_returns_false_when_stripe_not_configured(self, mock_subscription):
        """Test returns False when Stripe not configured."""
        from seer.api.subscriptions.overage_service import detach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config:
            mock_config.stripe_secret_key = None

            result = await detach_overage_pricing(mock_subscription, "si_123")

            assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_stripe_error(self, mock_subscription):
        """Test returns False on Stripe API error."""
        import stripe
        from seer.api.subscriptions.overage_service import detach_overage_pricing

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("stripe.SubscriptionItem.delete") as mock_delete:

            mock_config.stripe_secret_key = "sk_test_123"
            mock_delete.side_effect = stripe.error.StripeError("API Error")  # type: ignore[attr-defined]

            result = await detach_overage_pricing(mock_subscription, "si_123")

            assert result is False


# =============================================================================
# Report Usage To Stripe Tests
# =============================================================================


class TestReportUsageToStripe:
    """Tests for report_usage_to_stripe function (Billing Meter Events API)."""

    @pytest.mark.asyncio
    async def test_creates_meter_event_and_updates_records(
        self, mock_overage_settings, mock_billing_profile
    ):
        """Test creates Stripe meter event and updates local records."""
        from seer.api.subscriptions.overage_service import report_usage_to_stripe
        from seer.database.overage_models import OverageRecordStatus

        mock_overage_settings.billing_profile = mock_billing_profile
        mock_overage_settings.fetch_related = AsyncMock()

        mock_usage_record = MagicMock()
        mock_usage_record.save = AsyncMock()

        mock_meter_event = MagicMock()
        mock_meter_event.identifier = "evt_123"

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("seer.api.subscriptions.overage_service.OverageUsageRecord") as mock_record_cls, \
             patch("stripe.billing.MeterEvent.create") as mock_meter:

            mock_config.stripe_secret_key = "sk_test_123"
            mock_record_cls.create = AsyncMock(return_value=mock_usage_record)
            mock_meter.return_value = mock_meter_event

            result = await report_usage_to_stripe(
                overage_settings=mock_overage_settings,
                llm_record=None,
                base_cost_cents=100,
                billed_amount_cents=130,
            )

            assert result == mock_usage_record
            mock_meter.assert_called_once()
            call_kwargs = mock_meter.call_args.kwargs
            assert call_kwargs["event_name"] == "llm_overage_usage"
            assert call_kwargs["payload"]["value"] == "130"
            assert call_kwargs["payload"]["stripe_customer_id"] == "cus_test123"

            # Verify local record was updated
            assert mock_usage_record.status == OverageRecordStatus.REPORTED
            assert mock_usage_record.stripe_usage_record_id == "evt_123"

    @pytest.mark.asyncio
    async def test_returns_none_when_stripe_not_configured(self, mock_overage_settings):
        """Test returns None when Stripe not configured."""
        from seer.api.subscriptions.overage_service import report_usage_to_stripe

        with patch("seer.api.subscriptions.overage_service.config") as mock_config:
            mock_config.stripe_secret_key = None

            result = await report_usage_to_stripe(
                overage_settings=mock_overage_settings,
                llm_record=None,
                base_cost_cents=100,
                billed_amount_cents=130,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_customer_id(
        self, mock_overage_settings, mock_billing_profile
    ):
        """Test returns None when no Stripe customer ID."""
        from seer.api.subscriptions.overage_service import report_usage_to_stripe

        mock_billing_profile.stripe_customer_id = None
        mock_overage_settings.billing_profile = mock_billing_profile
        mock_overage_settings.fetch_related = AsyncMock()

        with patch("seer.api.subscriptions.overage_service.config") as mock_config:
            mock_config.stripe_secret_key = "sk_test_123"

            result = await report_usage_to_stripe(
                overage_settings=mock_overage_settings,
                llm_record=None,
                base_cost_cents=100,
                billed_amount_cents=130,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_marks_record_failed_on_stripe_error(
        self, mock_overage_settings, mock_billing_profile
    ):
        """Test marks usage record as failed on Stripe error."""
        import stripe
        from seer.api.subscriptions.overage_service import report_usage_to_stripe
        from seer.database.overage_models import OverageRecordStatus

        mock_overage_settings.billing_profile = mock_billing_profile
        mock_overage_settings.fetch_related = AsyncMock()

        mock_usage_record = MagicMock()
        mock_usage_record.save = AsyncMock()

        with patch("seer.api.subscriptions.overage_service.config") as mock_config, \
             patch("seer.api.subscriptions.overage_service.OverageUsageRecord") as mock_record_cls, \
             patch("stripe.billing.MeterEvent.create") as mock_meter:

            mock_config.stripe_secret_key = "sk_test_123"
            mock_record_cls.create = AsyncMock(return_value=mock_usage_record)
            mock_meter.side_effect = stripe.error.StripeError("Meter error")  # type: ignore[attr-defined]

            result = await report_usage_to_stripe(
                overage_settings=mock_overage_settings,
                llm_record=None,
                base_cost_cents=100,
                billed_amount_cents=130,
            )

            assert result == mock_usage_record
            assert mock_usage_record.status == OverageRecordStatus.FAILED
            assert "Meter error" in mock_usage_record.error_message


# =============================================================================
# Enable/Disable Overage Tests
# =============================================================================


class TestEnableOverage:
    """Tests for enable_overage function."""

    @pytest.mark.asyncio
    async def test_enables_overage_successfully(
        self, mock_billing_profile, mock_subscription, mock_overage_settings
    ):
        """Test enables overage and attaches Stripe pricing."""
        from seer.api.subscriptions.overage_service import enable_overage

        with patch(
            "seer.api.subscriptions.overage_service.is_overage_eligible"
        ) as mock_eligible, \
             patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings, \
             patch(
            "seer.api.subscriptions.overage_service.attach_overage_pricing"
        ) as mock_attach:

            mock_eligible.return_value = True
            mock_get_settings.return_value = mock_overage_settings
            mock_attach.return_value = "si_123"

            result = await enable_overage(mock_billing_profile, mock_subscription)

            assert result == mock_overage_settings
            assert mock_overage_settings.enabled is True
            assert mock_overage_settings.stripe_metered_subscription_item_id == "si_123"
            mock_overage_settings.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_not_eligible(
        self, mock_billing_profile, mock_subscription
    ):
        """Test raises ValueError when subscription not eligible."""
        from seer.api.subscriptions.overage_service import enable_overage

        with patch(
            "seer.api.subscriptions.overage_service.is_overage_eligible"
        ) as mock_eligible:
            mock_eligible.return_value = False

            with pytest.raises(ValueError) as exc_info:
                await enable_overage(mock_billing_profile, mock_subscription)

            assert "not eligible" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_stripe_attach_fails(
        self, mock_billing_profile, mock_subscription, mock_overage_settings
    ):
        """Test raises ValueError when Stripe attachment fails."""
        from seer.api.subscriptions.overage_service import enable_overage

        with patch(
            "seer.api.subscriptions.overage_service.is_overage_eligible"
        ) as mock_eligible, \
             patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings, \
             patch(
            "seer.api.subscriptions.overage_service.attach_overage_pricing"
        ) as mock_attach:

            mock_eligible.return_value = True
            mock_get_settings.return_value = mock_overage_settings
            mock_attach.return_value = None

            with pytest.raises(ValueError) as exc_info:
                await enable_overage(mock_billing_profile, mock_subscription)

            assert "Failed to attach" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_applies_custom_spending_cap(
        self, mock_billing_profile, mock_subscription, mock_overage_settings
    ):
        """Test applies custom spending cap within limits."""
        from seer.api.subscriptions.overage_service import enable_overage

        with patch(
            "seer.api.subscriptions.overage_service.is_overage_eligible"
        ) as mock_eligible, \
             patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings, \
             patch(
            "seer.api.subscriptions.overage_service.attach_overage_pricing"
        ) as mock_attach:

            mock_eligible.return_value = True
            mock_get_settings.return_value = mock_overage_settings
            mock_attach.return_value = "si_123"

            result = await enable_overage(
                mock_billing_profile, mock_subscription, spending_cap_cents=10000
            )

            assert result.spending_cap_cents == 10000


class TestDisableOverage:
    """Tests for disable_overage function."""

    @pytest.mark.asyncio
    async def test_disables_overage_and_detaches_stripe(
        self, mock_billing_profile, mock_subscription, mock_overage_settings
    ):
        """Test disables overage and detaches Stripe subscription item."""
        from seer.api.subscriptions.overage_service import disable_overage

        mock_overage_settings.enabled = True
        mock_overage_settings.stripe_metered_subscription_item_id = "si_123"

        with patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings, \
             patch(
            "seer.api.subscriptions.overage_service.detach_overage_pricing"
        ) as mock_detach:

            mock_get_settings.return_value = mock_overage_settings
            mock_detach.return_value = True

            result = await disable_overage(mock_billing_profile, mock_subscription)

            assert result == mock_overage_settings
            assert mock_overage_settings.enabled is False
            assert mock_overage_settings.stripe_metered_subscription_item_id is None
            mock_detach.assert_called_once_with(mock_subscription, "si_123")

    @pytest.mark.asyncio
    async def test_returns_settings_unchanged_if_already_disabled(
        self, mock_billing_profile, mock_subscription, mock_overage_settings
    ):
        """Test returns settings unchanged if already disabled."""
        from seer.api.subscriptions.overage_service import disable_overage

        mock_overage_settings.enabled = False

        with patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_overage_settings

            result = await disable_overage(mock_billing_profile, mock_subscription)

            assert result == mock_overage_settings
            mock_overage_settings.save.assert_not_called()


# =============================================================================
# Spending Cap Tests
# =============================================================================


class TestUpdateSpendingCap:
    """Tests for update_spending_cap function."""

    @pytest.mark.asyncio
    async def test_updates_spending_cap(
        self, mock_billing_profile, mock_overage_settings
    ):
        """Test updates spending cap successfully."""
        from seer.api.subscriptions.overage_service import update_spending_cap

        with patch(
            "seer.api.subscriptions.overage_service.get_or_create_overage_settings"
        ) as mock_get_settings:
            mock_get_settings.return_value = mock_overage_settings

            result = await update_spending_cap(mock_billing_profile, 10000)

            assert result.spending_cap_cents == 10000
            mock_overage_settings.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_cap_below_minimum(self, mock_billing_profile):
        """Test raises ValueError when cap below minimum."""
        from seer.api.subscriptions.overage_service import update_spending_cap

        with pytest.raises(ValueError) as exc_info:
            await update_spending_cap(mock_billing_profile, 100)  # $1 is below minimum

        assert "must be between" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_cap_above_maximum(self, mock_billing_profile):
        """Test raises ValueError when cap above maximum."""
        from seer.api.subscriptions.overage_service import update_spending_cap

        with pytest.raises(ValueError) as exc_info:
            await update_spending_cap(mock_billing_profile, 100000000)  # Too high

        assert "must be between" in str(exc_info.value)


# =============================================================================
# Period Reset Tests
# =============================================================================


class TestResetPeriodOverage:
    """Tests for reset_period_overage function."""

    @pytest.mark.asyncio
    async def test_resets_overage_counter(self, mock_overage_settings):
        """Test resets current period overage counter."""
        from seer.api.subscriptions.overage_service import reset_period_overage

        mock_overage_settings.current_period_overage_cents = 5000
        new_period_start = datetime(2026, 3, 1, tzinfo=timezone.utc)

        await reset_period_overage(mock_overage_settings, new_period_start)

        assert mock_overage_settings.current_period_overage_cents == 0
        assert mock_overage_settings.current_period_start == new_period_start
        mock_overage_settings.save.assert_called_once()


# =============================================================================
# Usage Summary Tests
# =============================================================================


class TestGetOverageUsageSummary:
    """Tests for get_overage_usage_summary function."""

    @pytest.mark.asyncio
    async def test_returns_usage_summary(self, mock_overage_settings):
        """Test returns comprehensive usage summary."""
        from seer.api.subscriptions.overage_service import get_overage_usage_summary

        mock_overage_settings.enabled = True
        mock_overage_settings.spending_cap_cents = 5000
        mock_overage_settings.spending_cap_dollars = Decimal("50.00")
        mock_overage_settings.current_period_overage_cents = 1500
        mock_overage_settings.current_period_overage_dollars = Decimal("15.00")
        mock_overage_settings.remaining_cap_cents = 3500
        mock_overage_settings.margin_multiplier = Decimal("1.30")
        mock_overage_settings.current_period_start = datetime(
            2026, 2, 1, tzinfo=timezone.utc
        )
        mock_overage_settings.enabled_at = datetime(2026, 1, 15, tzinfo=timezone.utc)
        mock_overage_settings.is_cap_reached = MagicMock(return_value=False)

        with patch(
            "seer.api.subscriptions.overage_service.OverageUsageRecord"
        ) as mock_record_cls:
            mock_filter = MagicMock()
            mock_filter.count = AsyncMock(side_effect=[2, 10, 1])  # pending, reported, failed
            mock_record_cls.filter.return_value = mock_filter

            result = await get_overage_usage_summary(mock_overage_settings)

            assert result["enabled"] is True
            assert result["spending_cap_cents"] == 5000
            assert result["current_usage_cents"] == 1500
            assert result["remaining_cents"] == 3500
            assert result["cap_reached"] is False
            assert result["records"]["pending"] == 2
            assert result["records"]["reported"] == 10
            assert result["records"]["failed"] == 1
