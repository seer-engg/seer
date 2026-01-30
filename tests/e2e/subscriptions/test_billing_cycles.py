# TODO: Fix these tests
# """
# E2E tests for monthly billing cycles after trial period.

# Tests recurring billing behavior:
# - First monthly charge after trial
# - Multiple billing cycles
# - Period date progression
# - Annual subscriptions
# """
# from datetime import datetime, timezone

# import pytest
# import stripe

# from seer.config import config
# from seer.database.subscription_models import BillingSubscription, SubscriptionStatus

# from .helpers import (
#     assert_invoice_amount,
#     assert_period_dates_progression,
#     assert_subscription_synced,
# )


# @pytest.mark.asyncio
# async def test_first_monthly_charge_after_trial(
#     trial_subscription_setup, stripe_test_clock, pro_monthly_price
# ):
#     """
#     Test that the first monthly charge occurs correctly after trial ends.

#     Verifies:
#     - Invoice created at trial end
#     - Amount is correct ($39 for Pro monthly)
#     - billing_reason is "subscription_cycle"
#     - Payment succeeded
#     - Subscription remains active
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     # Advance past trial
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for processing
#     import asyncio
#     await asyncio.sleep(2)

#     # Retrieve invoice
#     stripe.api_key = config.stripe_secret_key
#     invoices = stripe.Invoice.list(
#         subscription=stripe_subscription.id,
#         limit=1,
#     )

#     assert len(invoices.data) > 0, "Invoice should be created after trial"
#     invoice = invoices.data[0]

#     # Verify invoice details
#     assert invoice.billing_reason == "subscription_cycle"
#     assert invoice.status == "paid"
#     assert_invoice_amount(invoice, pro_monthly_price)

#     # Verify payment succeeded
#     assert invoice.payment_intent is not None
#     payment_intent = stripe.PaymentIntent.retrieve(invoice.payment_intent)
#     assert payment_intent.status == "succeeded"

#     # Verify subscription is active
#     updated_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     assert updated_sub.status == "active"

#     # Verify DB synced
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     await sync_subscription_from_stripe(updated_sub)

#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.ACTIVE
#     await assert_subscription_synced(subscription, stripe_subscription.id)


# @pytest.mark.asyncio
# async def test_second_monthly_billing_cycle(
#     trial_subscription_setup, stripe_test_clock, pro_monthly_price
# ):
#     """
#     ⭐ CRITICAL TEST: Verify recurring billing works for the second month.

#     Verifies:
#     - First charge at trial end (day 14)
#     - Second charge at day 44 (14 days trial + 30 days first cycle)
#     - Both invoices paid successfully
#     - Period dates updated correctly
#     - DB synced after each cycle
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     stripe.api_key = config.stripe_secret_key

#     # Get initial trial end
#     initial_sub = stripe.Subscription.retrieve(stripe_subscription.id)
#     trial_end = datetime.fromtimestamp(initial_sub.trial_end, tz=timezone.utc)

#     # Advance to trial end (first charge)
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for first charge
#     import asyncio
#     await asyncio.sleep(2)

#     # Verify first invoice
#     invoices_1 = stripe.Invoice.list(subscription=stripe_subscription.id, limit=1)
#     assert len(invoices_1.data) > 0
#     first_invoice = invoices_1.data[0]
#     assert first_invoice.status == "paid"
#     assert_invoice_amount(first_invoice, pro_monthly_price)

#     # Get period end after first charge
#     sub_after_first = stripe.Subscription.retrieve(stripe_subscription.id)
#     first_period_end = datetime.fromtimestamp(
#         sub_after_first.current_period_end, tz=timezone.utc
#     )

#     # Advance to second billing cycle (30 days after trial end)
#     stripe_test_clock.advance_clock(test_clock.id, days=30, hours=1)

#     # Wait for second charge
#     await asyncio.sleep(2)

#     # Verify second invoice
#     invoices_2 = stripe.Invoice.list(subscription=stripe_subscription.id, limit=2)
#     assert len(invoices_2.data) >= 2, "Second invoice should be created"

#     # Find the second invoice (most recent)
#     second_invoice = invoices_2.data[0]
#     if second_invoice.id == first_invoice.id:
#         second_invoice = invoices_2.data[1]

#     assert second_invoice.status == "paid"
#     assert_invoice_amount(second_invoice, pro_monthly_price)

#     # Verify period dates progressed
#     sub_after_second = stripe.Subscription.retrieve(stripe_subscription.id)
#     second_period_end = datetime.fromtimestamp(
#         sub_after_second.current_period_end, tz=timezone.utc
#     )

#     assert_period_dates_progression(
#         first_period_end, second_period_end, expected_interval_days=30
#     )

#     # Verify DB synced
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     await sync_subscription_from_stripe(sub_after_second)

#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.ACTIVE
#     await assert_subscription_synced(subscription, stripe_subscription.id)


# @pytest.mark.asyncio
# async def test_third_monthly_cycle_verification(
#     trial_subscription_setup, stripe_test_clock, pro_monthly_price
# ):
#     """
#     Test long-term stability by verifying 3 full billing cycles.

#     Verifies:
#     - All 3 cycles process successfully
#     - Each invoice is paid
#     - Period dates increment consistently
#     - No errors or inconsistencies
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     stripe.api_key = config.stripe_secret_key

#     # Advance through 3 full cycles: trial (14 days) + 3 months (90 days)
#     import asyncio

#     # Cycle 1: Trial end (day 14)
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)
#     await asyncio.sleep(2)

#     sub_1 = stripe.Subscription.retrieve(stripe_subscription.id)
#     period_end_1 = datetime.fromtimestamp(sub_1.current_period_end, tz=timezone.utc)

#     # Cycle 2: First month after trial (day 44)
#     stripe_test_clock.advance_clock(test_clock.id, days=30, hours=1)
#     await asyncio.sleep(2)

#     sub_2 = stripe.Subscription.retrieve(stripe_subscription.id)
#     period_end_2 = datetime.fromtimestamp(sub_2.current_period_end, tz=timezone.utc)

#     # Cycle 3: Second month (day 74)
#     stripe_test_clock.advance_clock(test_clock.id, days=30, hours=1)
#     await asyncio.sleep(2)

#     sub_3 = stripe.Subscription.retrieve(stripe_subscription.id)
#     period_end_3 = datetime.fromtimestamp(sub_3.current_period_end, tz=timezone.utc)

#     # Verify all 3 invoices
#     invoices = stripe.Invoice.list(subscription=stripe_subscription.id, limit=10)
#     paid_invoices = [inv for inv in invoices.data if inv.status == "paid"]

#     assert len(paid_invoices) >= 3, f"Expected 3 paid invoices, got {len(paid_invoices)}"

#     # Verify each invoice amount
#     for invoice in paid_invoices[:3]:
#         assert_invoice_amount(invoice, pro_monthly_price)

#     # Verify period date progression
#     assert_period_dates_progression(period_end_1, period_end_2, expected_interval_days=30)
#     assert_period_dates_progression(period_end_2, period_end_3, expected_interval_days=30)

#     # Verify subscription still active
#     assert sub_3.status == "active"

#     # Verify DB synced
#     from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#     await sync_subscription_from_stripe(sub_3)

#     await billing_profile.refresh_from_db()
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)
#     assert subscription.status == SubscriptionStatus.ACTIVE


# @pytest.mark.asyncio
# async def test_period_date_progression(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test that period dates increment correctly across multiple cycles.

#     Verifies:
#     - Each period_end increments by 30 days (monthly)
#     - DB dates match Stripe dates at each cycle
#     - No date drift or inconsistencies
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     stripe.api_key = config.stripe_secret_key
#     period_ends = []

#     # Track period ends across 3 cycles
#     import asyncio

#     for cycle in range(3):
#         # Advance to next cycle
#         if cycle == 0:
#             # First cycle: trial end
#             stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)
#         else:
#             # Subsequent cycles: 30 days
#             stripe_test_clock.advance_clock(test_clock.id, days=30, hours=1)

#         await asyncio.sleep(2)

#         # Get period end
#         sub = stripe.Subscription.retrieve(stripe_subscription.id)
#         period_end = datetime.fromtimestamp(sub.current_period_end, tz=timezone.utc)
#         period_ends.append(period_end)

#         # Sync to DB
#         from seer.api.subscriptions.stripe_service import sync_subscription_from_stripe
#         await sync_subscription_from_stripe(sub)

#         # Verify DB matches Stripe
#         await billing_profile.refresh_from_db()
#         subscription = await BillingSubscription.get(billing_profile=billing_profile)
#         db_period_end = subscription.current_period_end

#         if db_period_end:
#             diff = abs((db_period_end - period_end).total_seconds())
#             assert diff <= 1, f"Cycle {cycle + 1}: DB period end doesn't match Stripe (diff={diff}s)"

#     # Verify progression between cycles
#     assert_period_dates_progression(period_ends[0], period_ends[1], expected_interval_days=30)
#     assert_period_dates_progression(period_ends[1], period_ends[2], expected_interval_days=30)


# @pytest.mark.asyncio
# async def test_annual_subscription_after_trial(
#     user_with_payment_method, stripe_test_clock, pro_annual_price
# ):
#     """
#     Test annual subscription billing after trial.

#     Verifies:
#     - 14-day trial for annual subscription
#     - After trial, charge is annual amount ($390)
#     - Period is 365 days
#     - No monthly charges
#     """
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout
#     from .helpers import create_customer_with_test_clock, attach_test_payment_method

#     user, billing_profile, _ = user_with_payment_method

#     # Create test clock and customer
#     test_clock = stripe_test_clock.create_clock()
#     test_customer = create_customer_with_test_clock(
#         email=f"annual_billing_{user.email}",
#         test_clock_id=test_clock.id,
#     )
#     attach_test_payment_method(test_customer.id)

#     # Create annual subscription
#     stripe.api_key = config.stripe_secret_key
#     price_id = get_price_id_for_checkout("pro", "year", is_early_adopter=False)

#     subscription = stripe.Subscription.create(
#         customer=test_customer.id,
#         items=[{"price": price_id}],
#         trial_period_days=14,  # Start 14-day trial immediately
#         metadata={"user_id": user.user_id},
#     )

#     # Advance past trial
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     # Wait for processing
#     import asyncio
#     await asyncio.sleep(2)

#     # Verify annual charge
#     invoices = stripe.Invoice.list(subscription=subscription.id, limit=1)
#     assert len(invoices.data) > 0
#     invoice = invoices.data[0]

#     assert_invoice_amount(invoice, pro_annual_price)
#     assert invoice.status == "paid"

#     # Verify annual period (365 days)
#     updated_sub = stripe.Subscription.retrieve(subscription.id)
#     period_start = datetime.fromtimestamp(updated_sub.current_period_start, tz=timezone.utc)
#     period_end = datetime.fromtimestamp(updated_sub.current_period_end, tz=timezone.utc)

#     assert_period_dates_progression(period_start, period_end, expected_interval_days=365)

#     # Advance 30 days - should NOT create new invoice (annual subscription)
#     stripe_test_clock.advance_clock(test_clock.id, days=30)
#     await asyncio.sleep(2)

#     invoices_after = stripe.Invoice.list(subscription=subscription.id, limit=2)
#     # Should still be just the one annual invoice
#     assert len(invoices_after.data) == 1

#     # Cleanup
#     try:
#         stripe.Subscription.delete(subscription.id)
#         stripe.Customer.delete(test_customer.id)
#     except stripe.error.StripeError:
#         pass


# @pytest.mark.asyncio
# async def test_billing_cycle_with_proration(
#     trial_subscription_setup, stripe_test_clock
# ):
#     """
#     Test billing cycle behavior when user upgrades mid-cycle.

#     Verifies:
#     - User can upgrade from Pro to Pro+ mid-cycle
#     - Proration credit applied for unused time
#     - Next invoice includes both proration and new tier charge
#     """
#     user, billing_profile, stripe_subscription, test_clock = trial_subscription_setup

#     # Fetch subscription from DB
#     subscription = await BillingSubscription.get(billing_profile=billing_profile)

#     stripe.api_key = config.stripe_secret_key

#     # Let trial end and first cycle complete
#     stripe_test_clock.advance_clock(test_clock.id, days=14, hours=1)

#     import asyncio
#     await asyncio.sleep(2)

#     # Advance mid-cycle (15 days into first billing period)
#     stripe_test_clock.advance_clock(test_clock.id, days=15)

#     # Upgrade to Pro+ (simulating tier change)
#     from seer.api.subscriptions.pricing_catalog import get_price_id_for_checkout

#     pro_plus_price_id = get_price_id_for_checkout("pro_plus", "month", is_early_adopter=False)

#     # Update subscription to Pro+
#     updated_sub = stripe.Subscription.modify(
#         stripe_subscription.id,
#         items=[{
#             "id": stripe_subscription["items"]["data"][0]["id"],
#             "price": pro_plus_price_id,
#         }],
#         proration_behavior="create_prorations",
#     )

#     # Wait for proration invoice
#     await asyncio.sleep(2)

#     # Verify proration invoice created
#     invoices = stripe.Invoice.list(subscription=stripe_subscription.id, limit=2)

#     # Should have at least 2 invoices: original + proration
#     assert len(invoices.data) >= 2

#     # Verify subscription upgraded
#     assert updated_sub.status == "active"

#     # Note: Proration amounts are complex to calculate precisely,
#     # but we verified the upgrade mechanism works
