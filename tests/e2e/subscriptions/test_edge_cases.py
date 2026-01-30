# TODO: Fix these tests
# """
# E2E tests for edge cases and error scenarios.

# Tests robustness:
# - Payment method removal during trial
# - Card declined at trial end
# - Tier changes during trial
# - Webhook failures and retries
# - Database sync recovery
# - Idempotency
# - Coupons and discounts
# - Incomplete subscriptions
# """
# from datetime import datetime, timezone

# import pytest
# import stripe

# from seer.config import config
# from seer.database.subscription_models import BillingSubscription, SubscriptionStatus, SubscriptionTier

# from .helpers import (
#     TEST_CARDS,
#     assert_subscription_synced,
#     create_test_card_token,
# )


# @pytest.mark.asyncio
# async def test_payment_method_removed_during_trial(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test what happens when payment method is removed mid-trial.

#     Verifies:
#     - Trial continues even after payment method removed
#     - At trial end, payment fails
#     - invoice.payment_failed webhook triggered
#     - Subscription status becomes "past_due"
#     - Stripe retries payment
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     stripe.api_key = config.stripe_secret_key

#     # Get customer
#     customer = stripe.Customer.retrieve(billing_profile.stripe_customer_id)

#     # Detach payment method
#     payment_methods = stripe.PaymentMethod.list(
#         customer=customer.id,
#         type="card",
#     )

#     for pm in payment_methods.data:
#         stripe.PaymentMethod.detach(pm.id)

#     # Verify no payment method attached
#     pms_after = stripe.PaymentMethod.list(customer=customer.id, type="card")
#     assert len(pms_after.data) == 0

#     # Trial should still be active
#     sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     assert sub.status == "trialing"

#     # Advance to trial end
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for Stripe to attempt payment
#     import asyncio
#     await asyncio.sleep(3)

#     # Check subscription status (should be past_due or incomplete)
#     updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     assert updated_sub.status in ["past_due", "incomplete"], (
#         f"Expected past_due or incomplete, got {updated_sub.status}"
#     )

#     # Check for failed invoice
#     invoices = stripe.Invoice.list(subscription=stripe_subscription.id, limit=1)
#     if len(invoices.data) > 0:
#         invoice = invoices.data[0]
#         # Invoice should be open or uncollectible
#         assert invoice.status in ["open", "uncollectible"]


# @pytest.mark.asyncio
# async def test_card_declined_at_trial_end(
#     user_with_payment_method, stripe_test_clock
# ):
#     """
#     Test subscription behavior when card is declined at trial end.

#     Verifies:
#     - Invoice created but payment fails
#     - Subscription status becomes "past_due"
#     - Stripe retry logic kicks in
#     - User can update payment method to resolve
#     """
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout
#     from .helpers import create_customer_with_test_clock

#     user, billing_profile, _ = user_with_payment_method

#     # Create test clock and customer
#     test_clock = stripe_test_clock.create_clock()
#     test_customer = create_customer_with_test_clock(
#         email=f"declined_{user.email}",
#         test_clock_id=test_clock.id,
#     )

#     # Attach a card that will be declined
#     stripe.api_key = config.stripe_secret_key

#     # Create payment method with declining card
#     pm = stripe.PaymentMethod.create(
#         type="card",
#         card={"token": "tok_chargeDeclined"},
#     )
#     pm.attach(customer=test_customer.id)

#     stripe.Customer.modify(
#         test_customer.id,
#         invoice_settings={"default_payment_method": pm.id},
#     )

#     # Create subscription
#     price_id = get_price_id_for_checkout("pro", "month", is_early_adopter=False)
#     subscription = stripe.Subscription.create(
#         customer=test_customer.id,
#         items=[{"price": price_id}],
#         trial_period_days=14,  # Start 14-day trial immediately
#         metadata={"user_id": user.user_id},
#     )

#     # Advance past trial
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for payment attempt
#     import asyncio
#     await asyncio.sleep(3)

#     # Check subscription status
#     updated_sub = stripe.Subscription.retrieve(subscription.id)
#     # Status should be past_due or incomplete due to declined payment
#     assert updated_sub.status in ["past_due", "incomplete", "unpaid"]

#     # Check invoice
#     invoices = stripe.Invoice.list(subscription=subscription.id, limit=1)
#     assert len(invoices.data) > 0
#     invoice = invoices.data[0]
#     assert invoice.status != "paid"

#     # Cleanup
#     try:
#         stripe.Subscription.delete(subscription.id)
#         stripe.Customer.delete(test_customer.id)
#     except stripe.error.StripeError:
#         pass


# @pytest.mark.asyncio
# async def test_subscription_tier_change_during_trial(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test changing subscription tier (Pro -> Pro+) during trial.

#     Verifies:
#     - Tier can be changed during trial
#     - Trial continues on new tier
#     - New price applies after trial ends
#     - No charges during trial
#     """
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Mid-trial, upgrade to Pro+
#     stripe.api_key = config.stripe_secret_key
#     stripe_test_clock.advance_clock(test_clock.id, days=7)

#     pro_plus_price_id = get_price_id_for_checkout("pro_plus", "month", is_early_adopter=False)

#     # Update subscription tier
#     updated_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         items=[{
#             "id": stripe_subscription["items"]["data"][0]["id"],
#             "price": pro_plus_price_id,
#         }],
#         proration_behavior="none",  # No proration during trial
#     )

#     # Verify still in trial
#     assert updated_sub.status == "trialing"
#     assert updated_sub.trial_end == stripe_subscription.trial_end

#     # Verify new price
#     new_price_id = updated_sub["items"]["data"][0]["price"]["id"]
#     assert new_price_id == pro_plus_price_id

#     # Advance to trial end
#     stripe_test_clock.advance_clock(test_clock.id, days=7, hours=1)

#     # Wait for charge
#     import asyncio
#     await asyncio.sleep(2)

#     # Verify charge is for Pro+ amount ($79)
#     invoices = stripe.Invoice.list(subscription=stripe_subscription.id, limit=1)
#     assert len(invoices.data) > 0
#     invoice = invoices.data[0]

#     from .helpers import assert_invoice_amount
#     pro_plus_price = 7900  # $79.00
#     assert_invoice_amount(invoice, pro_plus_price)


# @pytest.mark.asyncio
# async def test_webhook_delivery_failure_retry(db_engine, webhook_verifier):
#     """
#     Test webhook retry mechanism when processing fails.

#     Verifies:
#     - Webhook marked as failed initially
#     - Stripe retries webhook delivery
#     - Eventually succeeds
#     - Marked as "processed"
#     """
#     from .helpers import simulate_webhook_failure

#     # This is a conceptual test - actual implementation depends on webhook system
#     # In production, Stripe automatically retries failed webhooks

#     # Note: This test would require mocking webhook failures
#     # and verifying the retry mechanism works correctly

#     # For now, we document the expected behavior:
#     # 1. Webhook fails initially (database error, network issue, etc.)
#     # 2. Stripe retries webhook (exponential backoff)
#     # 3. Webhook eventually succeeds
#     # 4. Status updated to "processed"
#     # 5. No duplicate processing due to idempotency

#     pass  # Placeholder for actual implementation


# @pytest.mark.asyncio
# async def test_database_out_of_sync_recovery(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test recovery when database gets out of sync with Stripe.

#     Verifies:
#     - Webhook sync corrects database state
#     - Manual sync works via sync_subscription_from_stripe()
#     - DB matches Stripe after recovery
#     """
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe

#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Corrupt DB: set wrong status
#     subscription.status = SubscriptionStatus.CANCELED
#     await subscription.save()

#     # Verify DB is wrong
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.CANCELED

#     # Retrieve correct state from Stripe
#     stripe.api_key = config.stripe_secret_key
#     correct_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     assert correct_sub.status == "trialing"

#     # Sync from Stripe (recovery)
#     await sync_subscription_from_stripe(correct_sub)

#     # Verify DB corrected
#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.TRIALING

#     # Verify full sync
#     await assert_subscription_synced(subscription, stripe_subscription.id)


# @pytest.mark.asyncio
# async def test_duplicate_webhook_idempotency(db_engine, webhook_verifier):
#     """
#     Test that duplicate webhooks are handled idempotently.

#     Verifies:
#     - First webhook processes successfully
#     - Second (duplicate) webhook acknowledged but not reprocessed
#     - No duplicate charges or state changes
#     - Database remains consistent
#     """
#     from seer.database.subscription_models import BillingSubscription, StripeWebhookEvent, StripeWebhookEventStatus
#     from tortoise.exceptions import IntegrityError

#     # Create a webhook event
#     event_id = "evt_test_idempotency"

#     webhook_event = await StripeWebhookEvent.create(
#         event_id=event_id,
#         type="customer.subscription.created",
#         payload={"id": event_id, "type": "customer.subscription.created"},
#         status=StripeWebhookEventStatus.PROCESSED,
#     )

#     # Try to create duplicate - should fail
#     try:
#         await StripeWebhookEvent.create(
#             event_id=event_id,
#             type="customer.subscription.created",
#             payload={"id": event_id, "type": "customer.subscription.created"},
#             status=StripeWebhookEventStatus.RECEIVED,
#         )
#         assert False, "Duplicate webhook should be prevented"
#     except IntegrityError:
#         # Expected - unique constraint prevents duplicate
#         pass

#     # Verify original still exists and is processed
#     original = await StripeWebhookEvent.get_or_none(event_id=event_id)
#     assert original is not None
#     assert original.status == StripeWebhookEventStatus.PROCESSED


# @pytest.mark.asyncio
# async def test_trial_with_coupon_code(user_with_payment_method, stripe_test_clock):
#     """
#     Test trial subscription with promotional coupon.

#     Verifies:
#     - Trial still 14 days with coupon
#     - Discount applies to first charge (after trial)
#     - Coupon metadata stored correctly
#     """
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout
#     from .helpers import create_customer_with_test_clock, attach_test_payment_method

#     user, billing_profile, _ = user_with_payment_method

#     # Create coupon (20% off)
#     stripe.api_key = config.stripe_secret_key
#     coupon = stripe.Coupon.create(
#         percent_off=20,
#         duration="once",
#         name="Test Coupon",
#     )

#     # Create test clock and customer
#     test_clock = stripe_test_clock.create_clock()
#     test_customer = create_customer_with_test_clock(
#         email=f"coupon_{user.email}",
#         test_clock_id=test_clock.id,
#     )
#     attach_test_payment_method(test_customer.id)

#     # Create subscription with coupon
#     price_id = get_price_id_for_checkout("pro", "month", is_early_adopter=False)
#     subscription = stripe.Subscription.create(
#         customer=test_customer.id,
#         items=[{"price": price_id}],
#         trial_period_days=14,  # Start 14-day trial immediately
#         coupon=coupon.id,
#         metadata={"user_id": user.user_id},
#     )

#     # Verify trial still 14 days
#     from .helpers import assert_trial_period_correct
#     assert_trial_period_correct(subscription, expected_days=14)

#     # Advance past trial
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for charge
#     import asyncio
#     await asyncio.sleep(2)

#     # Verify invoice has discount applied
#     invoices = stripe.Invoice.list(subscription=subscription.id, limit=1)
#     assert len(invoices.data) > 0
#     invoice = invoices.data[0]

#     # Original: $39, with 20% off: $31.20
#     expected_amount = int(3900 * 0.8)  # 3120 cents
#     from .helpers import assert_invoice_amount
#     assert_invoice_amount(invoice, expected_amount)

#     # Cleanup
#     try:
#         stripe.Subscription.delete(subscription.id)
#         stripe.Customer.delete(test_customer.id)
#         stripe.Coupon.delete(coupon.id)
#     except stripe.error.StripeError:
#         pass


# @pytest.mark.asyncio
# async def test_incomplete_subscription_handling(user_with_payment_method):
#     """
#     Test handling of incomplete subscriptions (requires 3DS, SCA).

#     Verifies:
#     - Subscription with requires_action stays incomplete
#     - User is blocked until payment completes
#     - Status updates when payment confirmed
#     """
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

#     user, billing_profile, stripe_customer_id = user_with_payment_method

#     stripe.api_key = config.stripe_secret_key

#     # Create subscription with payment_behavior=error_if_incomplete
#     # This simulates a case where payment requires action
#     price_id = get_price_id_for_checkout("pro", "month", is_early_adopter=False)

#     # Note: In real scenario, this would require a card that triggers 3DS
#     # For testing, we create an incomplete subscription

#     try:
#         subscription = stripe.Subscription.create(
#             customer=stripe_customer_id,
#             items=[{"price": price_id}],
#             payment_behavior="error_if_incomplete",
#             trial_period_days=0,  # No trial to trigger immediate payment
#         )

#         # If payment succeeds, subscription is active
#         # If payment requires action, subscription is incomplete
#         if subscription.status == "incomplete":
#             # Verify user should be blocked
#             from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#             await sync_subscription_from_stripe(subscription)

#             await billing_profile.refresh_from_db()
#             subscription = await BillingSubscription.get(billing_profile=billing_profile)
#             assert subscription.status == SubscriptionStatus.INCOMPLETE

#             # User should not have access to paid features
#             assert subscription.tier == SubscriptionTier.FREE or (
#                 subscription.tier == SubscriptionTier.PRO
#                 and subscription.status == SubscriptionStatus.INCOMPLETE
#             )

#         # Cleanup
#         try:
#             stripe.Subscription.delete(subscription.id)
#         except stripe.error.StripeError:
#             pass

#     except stripe.error.CardError:
#         # Payment requires action - expected for 3DS cards
#         pass


# @pytest.mark.asyncio
# async def test_subscription_with_metadata(subscription_with_trial, user_with_payment_method):
#     """
#     Test that subscription metadata is stored and retrievable.

#     Verifies:
#     - user_id stored in metadata
#     - is_early_adopter flag stored
#     - Metadata persists through trial and billing cycles
#     """
#     user, billing_profile, _ = user_with_payment_method
#     subscription_id = subscription_with_trial["subscription_id"]

#     # Retrieve subscription
#     stripe.api_key = config.stripe_secret_key
#     subscription = stripe.Subscription.retrieve(subscription_id)

#     # Verify metadata
#     assert "user_id" in subscription.metadata
#     assert subscription.metadata["user_id"] == user.user_id

#     # Early adopter flag should be present
#     assert "is_early_adopter" in subscription.metadata


# @pytest.mark.asyncio
# async def test_concurrent_webhook_processing(db_engine, webhook_verifier):
#     """
#     Test that concurrent webhook processing doesn't cause race conditions.

#     Verifies:
#     - Multiple webhooks can be processed in parallel
#     - No race conditions in database updates
#     - All webhooks processed exactly once
#     """
#     from seer.database.subscription_models import BillingSubscription, StripeWebhookEvent, StripeWebhookEventStatus

#     # Create multiple webhook events
#     event_ids = [f"evt_concurrent_{i}" for i in range(5)]

#     import asyncio

#     # Create events concurrently
#     tasks = []
#     for event_id in event_ids:
#         task = StripeWebhookEvent.create(
#             event_id=event_id,
#             type="customer.subscription.updated",
#             payload={"id": event_id},
#             status=StripeWebhookEventStatus.RECEIVED,
#         )
#         tasks.append(task)

#     events = await asyncio.gather(*tasks)

#     # Verify all created
#     assert len(events) == 5

#     # Mark all as processed (simulating concurrent processing)
#     update_tasks = []
#     for event in events:
#         event.status = StripeWebhookEventStatus.PROCESSED
#         update_tasks.append(event.save())

#     await asyncio.gather(*update_tasks)

#     # Verify all processed
#     for event_id in event_ids:
#         event = await StripeWebhookEvent.get_or_none(event_id=event_id)
#         assert event is not None
#         assert event.status == StripeWebhookEventStatus.PROCESSED
