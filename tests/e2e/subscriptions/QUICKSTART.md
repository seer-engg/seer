# 14-Day Free Trial - Quick Start Guide

## 🚀 Quick Test Guide

This guide will help you quickly verify the 14-day free trial implementation is working correctly.

## Prerequisites

```bash
# 1. Ensure Stripe test keys are configured
grep STRIPE .env

# Should show:
# STRIPE_SECRET_KEY=sk_test_...
# STRIPE_PUBLISHABLE_KEY=pk_test_...
# STRIPE_WEBHOOK_SECRET=whsec_...

# 2. Install dependencies
uv sync
```

## Option A: Run Automated Tests (Recommended)

### Run All Tests (5 minutes)

```bash
cd /home/lokesh/second/seer
uv run pytest tests/e2e/subscriptions/ -v
```

**Expected Result:** All 33 tests should pass ✅

### Run Critical Tests Only (2 minutes)

```bash
uv run pytest tests/e2e/subscriptions/ -k "critical" -v
```

**Expected Result:** 3 critical tests should pass ✅

### What Tests Verify

- ✅ Trial subscription created during onboarding
- ✅ Trial converts to active after 14 days
- ✅ First payment collected after trial
- ✅ Cancellation during trial prevents charges
- ✅ Monthly billing cycles work correctly
- ✅ Edge cases handled (payment failures, webhooks, etc.)

## Option B: Manual Testing (Frontend + Backend)

### Step 1: Start Backend

```bash
cd /home/lokesh/second/seer
uv run python src/seer/main.py
# Backend should start on http://localhost:8000
```

### Step 2: Start Frontend

```bash
cd /home/lokesh/second/seer-frontend
npm run dev
# Frontend should start on http://localhost:5173
```

### Step 3: Start Stripe Webhook Forwarding

```bash
# Terminal 3
stripe listen --forward-to http://localhost:8000/api/subscriptions/webhooks/stripe
```

### Step 4: Complete Onboarding

1. **Navigate:** Open http://localhost:5173
2. **Start Onboarding:** Create new account or clear onboarding state
3. **Step 1 - Discovery:** Select any option (e.g., "Reddit")
4. **Step 2 - Experience:** Select any option (e.g., "Well-versed")
5. **Step 3 - Integrations:** Select any or skip
6. **Step 4 - Plan Selection:** 🆕 **NEW STEP**
   - Choose **Pro** or **Pro+**
   - Choose **Monthly** or **Annual**
   - Click on pricing card to select
7. **Step 5 - Payment Method:**
   - Enter test card: `4242 4242 4242 4242`
   - Expiry: `12/30`
   - CVC: `123`
   - Click "Add Payment Method"

### Step 5: Verify Trial Created

**Expected Behavior:**

1. ✅ Success toast appears: "Welcome to Seer! Your 14-day free trial is now active. Trial ends [date]."
2. ✅ Redirected to home page
3. ✅ User has Pro/Pro+ features enabled

**Check Backend Logs:**

```bash
# Should see:
[INFO] Created trial subscription for user test_user: subscription_id=sub_..., tier=pro, status=trialing
```

**Check Database:**

```bash
cd /home/lokesh/second/seer
uv run python -c "
import asyncio
from seer.database import init_db
from seer.database.subscription_models import BillingProfile

async def check():
    await init_db()
    profiles = await BillingProfile.all()
    for p in profiles:
        print(f'User: {p.user_id}')
        print(f'Tier: {p.tier}')
        print(f'Status: {p.status}')
        print(f'Subscription ID: {p.stripe_subscription_id}')
        print(f'Trial End: {p.current_period_end}')
        print('---')

asyncio.run(check())
"
```

**Check Stripe Dashboard:**

1. Go to https://dashboard.stripe.com/test/customers
2. Find your test customer
3. Click on customer → Subscriptions tab
4. Verify:
   - Status: **Trialing**
   - Trial end: **14 days from now**
   - No invoices yet
   - No charges yet

## Option C: API Testing (Backend Only)

### Test Endpoint Directly

```bash
cd /home/lokesh/second/seer

# 1. Create test user with payment method
# (This would normally be done via auth flow)

# 2. Call create-with-trial endpoint
curl -X POST http://localhost:8000/api/subscriptions/create-with-trial \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TEST_TOKEN" \
  -d '{
    "tier": "pro",
    "interval": "month"
  }'

# Expected response:
# {
#   "subscription_id": "sub_...",
#   "status": "trialing",
#   "trial_end": "2026-02-12T..."
# }
```

### Verify in Database

```bash
# Connect to database
psql $DATABASE_URL

# Check subscription
SELECT
  user_id,
  tier,
  status,
  stripe_subscription_id,
  current_period_end
FROM billing_subscriptions
WHERE status = 'trialing';

# Expected result:
# user_id | tier | status   | stripe_subscription_id | current_period_end
# --------|------|----------|------------------------|-------------------
# test_123| PRO  | TRIALING | sub_xxx...            | 2026-02-12 ...
```

## Verification Checklist

After completing any of the above options, verify:

### ✅ Onboarding Flow

- [ ] Plan selection step appears as step 4
- [ ] Pricing cards show Pro and Pro+ options
- [ ] "14-day free trial" badge visible
- [ ] Monthly/Annual toggle works
- [ ] Payment method step appears as step 5

### ✅ Subscription Created

- [ ] Success toast displays with trial end date
- [ ] Backend logs show subscription creation
- [ ] Database has subscription with status=trialing
- [ ] Stripe dashboard shows trialing subscription

### ✅ Trial Behavior

- [ ] No charge created immediately
- [ ] Trial end date is 14 days from now
- [ ] User has access to paid features
- [ ] Subscription ID stored in database

### ✅ Webhooks

- [ ] Stripe webhook forwarding active (Terminal 3)
- [ ] `customer.subscription.created` webhook received
- [ ] Webhook processed successfully (check logs)

## Common Issues & Fixes

### Issue: "No payment method found"

**Cause:** User doesn't have payment method attached

**Fix:**
1. Verify Setup Intent completed successfully
2. Check Stripe customer has payment method attached
3. Ensure payment method saved before calling create-with-trial

### Issue: "Price not found"

**Cause:** Invalid tier or interval value

**Fix:**
1. Verify tier is "pro" or "pro_plus"
2. Verify interval is "month" or "year"
3. Check pricing_catalog.py has correct price IDs

### Issue: Tests failing with "Stripe API key not found"

**Cause:** Missing Stripe test keys in environment

**Fix:**
```bash
# Add to .env file
STRIPE_SECRET_KEY=sk_test_YOUR_KEY_HERE
STRIPE_PUBLISHABLE_KEY=pk_test_YOUR_KEY_HERE
```

### Issue: "Webhook signature verification failed"

**Cause:** Stripe CLI not running or wrong webhook secret

**Fix:**
```bash
# Terminal 1: Start Stripe CLI
stripe listen --forward-to http://localhost:8000/api/subscriptions/webhooks/stripe

# Terminal 2: Get webhook secret
stripe listen --print-secret

# Add to .env
STRIPE_WEBHOOK_SECRET=whsec_...
```

## Next Steps

Once you've verified the implementation works:

1. **Review Code:**
   - Backend: `/home/lokesh/second/seer/src/seer/api/subscriptions/router.py` (lines 343-457)
   - Frontend: `/home/lokesh/second/seer-frontend/src/pages/Onboarding.tsx`

2. **Read Full Documentation:**
   - Implementation Summary: `/home/lokesh/second/IMPLEMENTATION_SUMMARY.md`
   - Test Documentation: `/home/lokesh/second/seer/tests/e2e/subscriptions/README.md`

3. **Deploy to Staging:**
   - Push changes to staging branch
   - Run E2E tests in staging environment
   - Verify with real Stripe test mode webhooks

4. **Production Deployment:**
   - Code review and approval
   - Deploy backend endpoint first
   - Deploy frontend changes
   - Monitor Stripe dashboard and logs

## Quick Commands Reference

```bash
# Run all tests
uv run pytest tests/e2e/subscriptions/ -v

# Run critical tests only
uv run pytest tests/e2e/subscriptions/ -k "critical" -v

# Run specific test file
uv run pytest tests/e2e/subscriptions/test_trial_expiration.py -v

# Run with coverage
uv run pytest tests/e2e/subscriptions/ --cov=src/seer/api/subscriptions --cov-report=html

# Start backend
cd /home/lokesh/second/seer && uv run python src/seer/main.py

# Start frontend
cd /home/lokesh/second/seer-frontend && npm run dev

# Start Stripe webhook forwarding
stripe listen --forward-to http://localhost:8000/api/subscriptions/webhooks/stripe

# Check database subscriptions
psql $DATABASE_URL -c "SELECT user_id, tier, status FROM billing_subscriptions;"
```

## Support

- **Tests failing?** Check `/home/lokesh/second/seer/tests/e2e/subscriptions/README.md`
- **API errors?** Check backend logs with `uv run python src/seer/main.py`
- **Stripe issues?** Check dashboard at https://dashboard.stripe.com/test
- **Frontend errors?** Check browser console and network tab

---

**Quick Start Version:** 1.0
**Last Updated:** 2026-01-29
