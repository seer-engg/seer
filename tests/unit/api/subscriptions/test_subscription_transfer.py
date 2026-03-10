"""
Unit tests for subscription transfer between organizations.

Tests cover:
- transfer_subscription_between_orgs function (org-centric model)
- Stripe customer FK transfer between orgs
- Source org reset to FREE tier
- Stripe customer metadata updates
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from seer.database.organization_models import Organization, OrganizationType
from seer.database.overage_models import OverageSettings
from seer.database.subscription_models import (
    BillingSubscription,
    StripeCustomer,
    SubscriptionStatus,
    SubscriptionTier,
)


pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def mock_transaction():
    """Mock async context manager for in_transaction()."""
    # Return a mock connection object
    yield MagicMock()


# =============================================================================
# Organization-to-Organization Transfer Tests
# =============================================================================


class TestTransferSubscriptionBetweenOrgs:
    """Tests for transfer_subscription_between_orgs function."""

    async def test_transfers_stripe_customer_to_target_org(self, mocker):
        """Transfer moves stripe_customer FK from source to target org."""
        from seer.api.subscriptions.stripe_service import transfer_subscription_between_orgs

        # Create source org with stripe customer
        source_org = MagicMock(spec=Organization)
        source_org.id = 1
        source_org.stripe_customer_id = 10
        source_org.has_payment_method = True
        source_org.payment_method_added_at = "2024-01-01"
        source_org.type = OrganizationType.PERSONAL

        target_org = MagicMock(spec=Organization)
        target_org.id = 2
        target_org.stripe_customer_id = None
        target_org.has_payment_method = False
        target_org.type = OrganizationType.TEAM
        target_org.name = "Test Team"

        stripe_customer = MagicMock(spec=StripeCustomer)
        stripe_customer.id = 10
        stripe_customer.stripe_customer_id = "cus_source123"

        source_subscription = MagicMock(spec=BillingSubscription)
        source_subscription.organization = source_org
        source_subscription.tier = SubscriptionTier.PRO
        source_subscription.save = AsyncMock()

        # Mock database lookups
        mocker.patch.object(
            BillingSubscription, "get_or_none",
            new_callable=AsyncMock,
            return_value=source_subscription,
        )
        mocker.patch.object(
            StripeCustomer, "get",
            new_callable=AsyncMock,
            return_value=stripe_customer,
        )
        mocker.patch.object(
            OverageSettings, "get_or_none",
            new_callable=AsyncMock,
            return_value=None,
        )
        mocker.patch.object(
            BillingSubscription, "create",
            new_callable=AsyncMock,
        )

        # Mock filter().using_db().delete() chain for deleting target org's subscription
        mock_filter_chain = MagicMock()
        mock_filter_chain.using_db.return_value.delete = AsyncMock()
        mocker.patch.object(
            BillingSubscription, "filter",
            return_value=mock_filter_chain,
        )

        # Mock the transaction context manager
        mocker.patch(
            "seer.api.subscriptions.stripe_service.in_transaction",
            side_effect=mock_transaction,
        )

        # Mock select_for_update to return the mocked orgs
        locked_source = MagicMock(spec=Organization)
        locked_source.id = 1
        locked_source.stripe_customer_id = 10
        locked_source.has_payment_method = True
        locked_source.payment_method_added_at = "2024-01-01"
        locked_source.save = AsyncMock()

        locked_target = MagicMock(spec=Organization)
        locked_target.id = 2
        locked_target.save = AsyncMock()

        mock_query = MagicMock()
        mock_query.using_db.return_value.get = AsyncMock(side_effect=[locked_source, locked_target])
        mocker.patch.object(Organization, "select_for_update", return_value=mock_query)

        with patch("seer.api.subscriptions.stripe_service.stripe") as mock_stripe:
            await transfer_subscription_between_orgs(source_org, target_org)

        # Verify target org received the stripe_customer_id
        assert locked_target.stripe_customer_id == 10
        assert locked_target.has_payment_method is True

        # Verify source org was cleared
        assert locked_source.stripe_customer_id is None
        assert locked_source.has_payment_method is False

    async def test_raises_error_when_source_has_no_stripe_customer(self, mocker):
        """Transfer raises ValueError when source org has no stripe customer."""
        from seer.api.subscriptions.stripe_service import transfer_subscription_between_orgs

        source_org = MagicMock(spec=Organization)
        source_org.id = 1
        source_org.stripe_customer_id = None  # No stripe customer

        target_org = MagicMock(spec=Organization)
        target_org.id = 2

        with pytest.raises(ValueError, match="has no Stripe customer"):
            await transfer_subscription_between_orgs(source_org, target_org)

    async def test_raises_error_when_source_has_no_subscription(self, mocker):
        """Transfer raises ValueError when source org has no subscription."""
        from seer.api.subscriptions.stripe_service import transfer_subscription_between_orgs

        source_org = MagicMock(spec=Organization)
        source_org.id = 1
        source_org.stripe_customer_id = 10

        target_org = MagicMock(spec=Organization)
        target_org.id = 2

        mocker.patch.object(
            BillingSubscription, "get_or_none",
            new_callable=AsyncMock,
            return_value=None,  # No subscription
        )

        with pytest.raises(ValueError, match="has no subscription"):
            await transfer_subscription_between_orgs(source_org, target_org)

    async def test_creates_free_subscription_for_source_org(self, mocker):
        """Transfer creates FREE subscription for source org after moving."""
        from seer.api.subscriptions.stripe_service import transfer_subscription_between_orgs

        source_org = MagicMock(spec=Organization)
        source_org.id = 1
        source_org.stripe_customer_id = 10
        source_org.has_payment_method = True
        source_org.payment_method_added_at = "2024-01-01"
        source_org.type = OrganizationType.PERSONAL

        target_org = MagicMock(spec=Organization)
        target_org.id = 2
        target_org.type = OrganizationType.TEAM
        target_org.name = "Test Team"

        stripe_customer = MagicMock(spec=StripeCustomer)
        stripe_customer.id = 10
        stripe_customer.stripe_customer_id = "cus_source123"

        source_subscription = MagicMock(spec=BillingSubscription)
        source_subscription.organization = source_org
        source_subscription.tier = SubscriptionTier.PRO
        source_subscription.save = AsyncMock()

        mocker.patch.object(
            BillingSubscription, "get_or_none",
            new_callable=AsyncMock,
            return_value=source_subscription,
        )
        mocker.patch.object(
            StripeCustomer, "get",
            new_callable=AsyncMock,
            return_value=stripe_customer,
        )
        mocker.patch.object(
            OverageSettings, "get_or_none",
            new_callable=AsyncMock,
            return_value=None,
        )

        mock_create_sub = mocker.patch.object(
            BillingSubscription, "create",
            new_callable=AsyncMock,
        )

        # Mock filter().using_db().delete() chain for deleting target org's subscription
        mock_filter_chain = MagicMock()
        mock_filter_chain.using_db.return_value.delete = AsyncMock()
        mocker.patch.object(
            BillingSubscription, "filter",
            return_value=mock_filter_chain,
        )

        mocker.patch(
            "seer.api.subscriptions.stripe_service.in_transaction",
            side_effect=mock_transaction,
        )

        locked_source = MagicMock(spec=Organization)
        locked_source.id = 1
        locked_source.stripe_customer_id = 10
        locked_source.has_payment_method = True
        locked_source.payment_method_added_at = "2024-01-01"
        locked_source.save = AsyncMock()

        locked_target = MagicMock(spec=Organization)
        locked_target.id = 2
        locked_target.save = AsyncMock()

        mock_query = MagicMock()
        mock_query.using_db.return_value.get = AsyncMock(side_effect=[locked_source, locked_target])
        mocker.patch.object(Organization, "select_for_update", return_value=mock_query)

        with patch("seer.api.subscriptions.stripe_service.stripe"):
            await transfer_subscription_between_orgs(source_org, target_org)

        # Verify FREE subscription was created for source
        create_call = mock_create_sub.call_args
        assert create_call.kwargs["organization"] == locked_source
        assert create_call.kwargs["tier"] == SubscriptionTier.FREE
        assert create_call.kwargs["status"] == SubscriptionStatus.ACTIVE
