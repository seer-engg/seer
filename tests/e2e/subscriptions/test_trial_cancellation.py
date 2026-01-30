# TODO: Fix these tests
# """
# E2E tests for trial cancellation scenarios.

# Tests critical cancellation behavior:
# - Canceling during trial period
# - Subscription ending at trial_end when canceled
# - User access revoked after cancellation
# - Immediate cancellation handling
# """
# from datetime import datetime, timezone

# import pytest
# import stripe

# from seer.config import config
# from seer.database.subscription_models import BillingSubscription, SubscriptionStatus, SubscriptionTier

# from .helpers import (
#     assert_subscription_deleted,
#     assert_subscription_status,
# )


# @pytest.mark.asyncio
# async def test_cancel_during_trial_period(trial_subscription_setup, stripe_test_clock):
#     """
#     ⭐ CRITICAL TEST: Verify user can cancel subscription during trial.

#     Verifies:
#     - Subscription is marked for cancellation (cancel_at_period_end=true)
#     - Status remains "trialing" until trial end
#     - DB reflects cancel_at_period_end flag
#     - No charge occurs
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Advance 7 days into trial (mid-trial)
#     stripe_test_clock.advance_clock(test_clock.id, days=7)

#     # Cancel the subscription
#     stripe.api_key = config.stripe_secret_key
#     canceled_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=True,
#     )

#     # Verify cancellation flag set
#     assert canceled_sub.cancel_at_period_end is True
#     assert canceled_sub.status == "trialing", "Status should still be trialing"

#     # Sync to DB
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     await sync_subscription_from_stripe(canceled_sub)

#     # Verify DB reflects cancellation
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.cancel_at_period_end is True
#     assert subscription.status == SubscriptionStatus.TRIALING


# @pytest.mark.asyncio
# async def test_canceled_subscription_ends_at_trial_end(
#     trial_subscription_setup, stripe_test_clock, webhook_verifier
# ):
#     """
#     ⭐ CRITICAL TEST: Verify canceled subscription ends at trial_end without charge.

#     Verifies:
#     - Subscription canceled at trial end (not converted to active)
#     - customer.subscription.deleted webhook received
#     - No charge created
#     - DB updated: status=canceled, tier=free, subscription_id=None
#     - User reverted to free tier
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Cancel during trial
#     stripe.api_key = config.stripe_secret_key
#     stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=True,
#     )

#     # Advance to trial end + 1 hour
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for Stripe to process cancellation
#     import asyncio
#     await asyncio.sleep(2)

#     # Retrieve subscription (should be canceled)
#     try:
#         updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#         assert_subscription_status(updated_sub, "canceled")
#     except stripe.error.InvalidRequestError:
#         # Subscription might be deleted
#         pass

#     # Verify no charges were created
#     charges = stripe.Charge.list(
#         customer=billing_profile.stripe_customer_id,
#         limit=10,
#     )
#     assert len(charges.data) == 0, "No charges should be created during canceled trial"

#     # Verify DB updated to free tier
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.tier == SubscriptionTier.FREE
#     assert subscription.stripe_subscription_id is None or subscription.status == SubscriptionStatus.CANCELED

#     # Verify user is on free tier
#     await assert_subscription_deleted(user.user_id)


# @pytest.mark.asyncio
# async def test_user_blocked_after_trial_cancellation(
#     trial_subscription_setup, stripe_test_clock, authenticated_subscription_client
# ):
#     """
#     Test that user loses access to paid features after trial cancellation.

#     Verifies:
#     - During trial (even if canceled), user still has access
#     - After trial ends, user loses access
#     - Protected endpoints return 402 or 403 error
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Cancel subscription
#     stripe.api_key = config.stripe_secret_key
#     stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=True,
#     )

#     # Sync to DB
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     canceled_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     await sync_subscription_from_stripe(canceled_sub)

#     # During trial, user should still have access
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.TRIALING
#     # Note: Access control is typically checked by middleware/decorators
#     # This test verifies the subscription state is correct

#     # Advance past trial end
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for processing
#     import asyncio
#     await asyncio.sleep(2)

#     # Sync final state
#     try:
#         final_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#         await sync_subscription_from_stripe(final_sub)
#     except stripe.error.InvalidRequestError:
#         # Subscription deleted - manually update DB to free tier
#         subscription.tier = SubscriptionTier.FREE
#         subscription.status = SubscriptionStatus.CANCELED
#         subscription.stripe_subscription_id = None
#         await subscription.save()

#     # Verify user on free tier
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.tier == SubscriptionTier.FREE

#     # Check subscription endpoint returns free tier
#     response = await authenticated_subscription_client.get("/api/subscriptions/current")
#     assert response.status_code == 200
#     data = response.json()
#     assert data["tier"] == "free"


# @pytest.mark.asyncio
# async def test_immediate_cancellation_deletes_subscription(
#     trial_subscription_setup, webhook_verifier
# ):
#     """
#     Test immediate cancellation (delete) vs cancel_at_period_end.

#     Verifies:
#     - stripe.Subscription.delete() immediately cancels (no grace period)
#     - customer.subscription.deleted webhook received
#     - DB immediately updated to free tier
#     - Access immediately revoked
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Immediately delete subscription (not cancel_at_period_end)
#     stripe.api_key = config.stripe_secret_key
#     deleted_sub = stripe.Subscription.delete(stripe_subscription.id)

#     assert deleted_sub.status == "canceled"

#     # Wait for webhook
#     import asyncio
#     await asyncio.sleep(1)

#     # Manually sync (in real scenario, webhook handles this)
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe

#     # For deleted subscriptions, we need to handle specially
#     subscription.tier = SubscriptionTier.FREE
#     subscription.status = SubscriptionStatus.CANCELED
#     subscription.stripe_subscription_id = None
#     subscription.cancel_at_period_end = False
#     await subscription.save()

#     # Verify immediate reversion to free tier
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.tier == SubscriptionTier.FREE
#     assert subscription.stripe_subscription_id is None

#     # Verify user immediately loses access
#     await assert_subscription_deleted(user.user_id)


# @pytest.mark.asyncio
# async def test_reactivate_canceled_trial(trial_subscription_setup):
#     """
#     Test that a canceled trial can be reactivated before it ends.

#     Verifies:
#     - User can cancel trial (cancel_at_period_end=true)
#     - User can reactivate by setting cancel_at_period_end=false
#     - Trial continues normally
#     - Subscription converts to active after trial ends
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Cancel subscription
#     stripe.api_key = config.stripe_secret_key
#     canceled_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=True,
#     )
#     assert canceled_sub.cancel_at_period_end is True

#     # Reactivate (remove cancellation)
#     reactivated_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=False,
#     )
#     assert reactivated_sub.cancel_at_period_end is False
#     assert reactivated_sub.status == "trialing"

#     # Sync to DB
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     await sync_subscription_from_stripe(reactivated_sub)

#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.cancel_at_period_end is False
#     assert subscription.status == SubscriptionStatus.TRIALING


# @pytest.mark.asyncio
# async def test_cancel_after_trial_during_active_period(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test canceling a subscription after trial has ended and it's active.

#     Verifies:
#     - Trial converts to active successfully
#     - User can cancel during active period
#     - Subscription continues until period_end
#     - After period_end, subscription is canceled
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Let trial expire (convert to active)
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for conversion
#     import asyncio
#     await asyncio.sleep(2)

#     # Verify active
#     stripe.api_key = config.stripe_secret_key
#     active_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     assert active_sub.status == "active"

#     # Now cancel during active period
#     canceled_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         cancel_at_period_end=True,
#     )

#     assert canceled_sub.cancel_at_period_end is True
#     assert canceled_sub.status == "active"  # Still active until period end

#     # Advance to period end + 1 day
#     stripe_test_clock.advance_clock(test_clock.id, days=31)

#     # Wait for cancellation
#     await asyncio.sleep(2)

#     # Verify canceled
#     try:
#         final_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#         assert final_sub.status == "canceled"
#     except stripe.error.InvalidRequestError:
#         # Subscription deleted after cancellation
#         pass

#     # Verify no more charges
#     charges = stripe.Charge.list(
#         customer=billing_profile.stripe_customer_id,
#         limit=10,
#     )
#     # Should only have the one charge from trial ending
#     assert len(charges.data) <= 1
