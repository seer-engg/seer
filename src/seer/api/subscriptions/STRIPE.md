# Stripe Setup Guide

This document describes how to configure Stripe products, prices, and webhooks for Seer's subscription and overage billing system.

## Overview

**Stripe is the single source of truth for pricing.** Products and prices are:
- Fetched dynamically from Stripe via the API
- Cached in-memory with a 10-minute TTL (see `pricing_catalog.py:71`)
- Never hardcoded in the codebase

This means you must configure products and prices in the Stripe Dashboard before the application can serve pricing data.

---

## Products to Create

Create **2 subscription products** in Stripe Dashboard > Product Catalog.

### Lite Product

| Field | Value |
|-------|-------|
| Name | Lite |
| **Metadata** | |
| `tier` | `lite` |
| `display_name` | `Lite` |
| `features` | `["10 workflows", "500K monthly runs", "5-minute polling", "$10 LLM credits/month"]` |
| `sort_order` | `1` |

### Pro Product

| Field | Value |
|-------|-------|
| Name | Pro |
| **Metadata** | |
| `tier` | `pro` |
| `display_name` | `Pro` |
| `features` | `["Unlimited workflows", "1M monthly runs", "1-minute polling", "$20 LLM credits/month"]` |
| `sort_order` | `2` |
| `badge` *(optional)* | `MOST POPULAR` |

### Product Metadata Reference

| Key | Required | Description |
|-----|----------|-------------|
| `tier` | Yes | Internal tier identifier: `lite`, `pro` |
| `display_name` | Yes | UI display name |
| `features` | Yes | JSON array of feature strings |
| `sort_order` | No | Display order (lower = first) |
| `badge` | No | Badge text (e.g., "MOST POPULAR") |
| `upgrade_benefits` | No | JSON array for upgrade modal |

---

## Prices to Create

For each product above, create **2 prices** (monthly + annual = 4 total prices).

### Monthly Price Template

| Field | Value |
|-------|-------|
| Billing period | Recurring (Monthly) |
| **Metadata** | |
| `tier` | `<tier_name>` (e.g., `lite`, `pro`) |
| `variant` | `regular` |
| `trial_period_days` *(optional)* | `14` |

### Annual Price Template

| Field | Value |
|-------|-------|
| Billing period | Recurring (Yearly) |
| **Metadata** | |
| `tier` | `<tier_name>` |
| `variant` | `regular` |
| `original_price_cents` *(optional)* | Monthly price × 12 (for strikethrough display) |

### Price Metadata Reference

| Key | Required | Description |
|-----|----------|-------------|
| `tier` | Yes | Must match product tier for webhook resolution |
| `variant` | Yes | `regular` (or `early_adopter` for legacy) |
| `trial_period_days` | No | Free trial length |
| `original_price_cents` | No | For displaying discounted prices |

---

## Overage Billing with Stripe Meters

Stripe requires **Billing Meters** for usage-based pricing. This involves two steps:
1. Create a Meter (defines how usage is aggregated)
2. Create a Price linked to that Meter

### Step 1: Create the Billing Meter

Go to **Stripe Dashboard > Billing > Meters** and create a new meter:

| Field | Value | Notes |
|-------|-------|-------|
| **Display name** | `LLM Credit Overage` | Human-readable name |
| **Event name** | `llm_overage_usage` | **Critical:** Must match code constant |
| **Aggregation formula** | `SUM` | Sums all usage events in period |
| **Value key** | `value` | Key in event payload containing amount |

> **Important:** The event name `llm_overage_usage` must exactly match the constant in `overage_service.py:38`.

### Step 2: Create the Metered Price

After creating the meter, create a price linked to it:

| Field | Value |
|-------|-------|
| **Product** | Create new: "LLM Overage Credits" |
| **Pricing model** | Usage-based |
| **Meter** | Select the meter you just created |
| **Price per unit** | `$0.01` (1 cent per unit) |
| **Billing period** | Monthly |
| **Lookup key** | `llm_overage_metered` |
| **Metadata** | |
| `type` | `overage` |
| `purpose` | `llm_overage` |

### How Overage Works

```
┌──────────────────────────────────────────────────────────────────┐
│  User exceeds LLM credit limit                                    │
│                    ↓                                              │
│  overage_service.py calls stripe.billing.MeterEvent.create()     │
│      - event_name: "llm_overage_usage"                           │
│      - payload: { value: <cents>, stripe_customer_id: <id> }     │
│                    ↓                                              │
│  Stripe aggregates all meter events for the billing period       │
│                    ↓                                              │
│  End of period: Stripe generates invoice with usage charges      │
└──────────────────────────────────────────────────────────────────┘
```

### How Overage Prices Are Identified

The system identifies overage prices using this logic (from `pricing_catalog.py:253-276`):

1. `recurring.usage_type == "metered"`
2. AND either:
   - `lookup_key` contains `"overage"`, OR
   - `metadata.type == "overage"`

---

## Credit Limits Reference

These are the default LLM credit limits per tier (from `constants.py`):

| Tier | Monthly | 5-Hour Window | Weekly Window | Min Polling Interval |
|------|---------|---------------|---------------|----------------------|
| Free | $1.00 | $1.00 | $1.00 | 15 minutes |
| Lite | $10.00 | $3.00 | $6.00 | 5 minutes |
| Pro | $20.00 | $5.00 | $12.00 | 1 minute |

---

## Overage Configuration

Default overage settings (from `constants.py:103-119`):

| Setting | Value | Description |
|---------|-------|-------------|
| Margin multiplier | 1.30 | 30% markup on LLM cost |
| Minimum cap | $5.00 | Minimum spending cap users can set |
| Maximum cap | $1,000.00 | Maximum spending cap |
| Default cap | $50.00 | Default cap for new users |
| Warning threshold | 80% | Warn when approaching cap |

---

## Webhook Configuration

### Endpoint

Configure your webhook in Stripe Dashboard > Developers > Webhooks:

```
POST /subscriptions/webhooks/stripe
```

### Required Events

Select these events for the webhook endpoint:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `invoice.paid`
- `invoice.payment_failed`
- `setup_intent.succeeded`

### Event Handling

| Event | Handler |
|-------|---------|
| `checkout.session.completed` | Syncs Stripe customer to Clerk, syncs subscription |
| `customer.subscription.created/updated` | Syncs subscription tier and status |
| `customer.subscription.deleted` | Handles deletion, cleans up overage settings |
| `invoice.paid` | Resets overage counter for new billing period |
| `invoice.payment_failed` | Logs warning, may disable overage if overage-related |
| `setup_intent.succeeded` | Updates `has_payment_method` flag on billing profile |

---

## Environment Variables

Set these in your `.env` file:

```bash
STRIPE_SECRET_KEY=sk_live_...      # or sk_test_... for test mode
STRIPE_PUBLISHABLE_KEY=pk_live_... # or pk_test_... for test mode
STRIPE_WEBHOOK_SECRET=whsec_...    # From webhook endpoint settings
```

---

## Test Mode Setup Checklist

Use this checklist when setting up Stripe in test mode:

### Products
- [ ] Create Lite product with `tier: lite` metadata
- [ ] Create Pro product with `tier: pro` metadata

### Prices
- [ ] Create monthly price for Lite (`tier: lite`, `variant: regular`)
- [ ] Create annual price for Lite
- [ ] Create monthly price for Pro (`tier: pro`, `variant: regular`)
- [ ] Create annual price for Pro

### Overage (Billing Meter)
- [ ] Create Billing Meter with event name `llm_overage_usage`
- [ ] Create metered price linked to the meter
- [ ] Set `lookup_key: llm_overage_metered` on the price

### Webhook
- [ ] Configure webhook endpoint URL
- [ ] Select required events
- [ ] Copy webhook signing secret to `STRIPE_WEBHOOK_SECRET`

### Environment
- [ ] Set `STRIPE_SECRET_KEY`
- [ ] Set `STRIPE_PUBLISHABLE_KEY`
- [ ] Set `STRIPE_WEBHOOK_SECRET`

---

## Verification

After configuration, verify your setup:

1. **Start/restart the application**

2. **Check pricing endpoint:**
   ```bash
   curl http://localhost:8000/subscriptions/pricing
   ```
   Should return all tiers with their prices.

3. **Check logs for cache refresh:**
   ```
   Pricing cache refreshed: 2 products, 4 prices, overage_price=price_xxx
   ```

4. **Test checkout flow:**
   - Create a checkout session
   - Complete test payment
   - Verify tier assignment

5. **Test overage flow:**
   - Enable overage for a user
   - Verify metered subscription item is created
   - Test usage reporting

---

## Key Source Files

| File | Purpose |
|------|---------|
| `pricing_catalog.py` | Dynamic pricing fetch and caching |
| `stripe_service.py` | Stripe API interactions |
| `overage_service.py` | Overage management and usage reporting |
| `stripe_webhook_controller.py` | Webhook event handling |
| `constants.py` (observability) | Credit limits and overage constants |

---

## Troubleshooting

### "Pricing cache empty" warning
- Check that `STRIPE_SECRET_KEY` is set
- Verify products have `tier` metadata

### Tier not appearing in pricing
- Ensure product has required metadata: `tier`, `display_name`, `features`
- Ensure both monthly AND annual prices exist with `tier` metadata

### Webhook events not processing
- Verify `STRIPE_WEBHOOK_SECRET` matches the webhook endpoint
- Check webhook endpoint URL is accessible
- Review webhook logs in Stripe Dashboard

### Overage price not detected
- Ensure price has `usage_type: metered` in recurring settings
- Verify `lookup_key` contains "overage" OR `metadata.type = "overage"`
