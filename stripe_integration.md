# Stripe Integration Plan

## Overview

Integrate Stripe subscriptions into the Seer workflow builder app with:
- **4 tiers**: Free, Pro ($20), Pro+ ($60), Ultra ($100)
- **Billing cycles**: Monthly and Annual (with ~17% discount for annual)
- **Stripe Checkout**: Hosted payment pages (no custom payment forms)
- **Stripe Customer Portal**: Self-service subscription management
- **Clerk Sync**: Store Stripe customer ID in Clerk user metadata

---

## Table of Contents

1. [Stripe Dashboard Setup](#1-stripe-dashboard-setup)
2. [Backend Implementation](#2-backend-implementation)
3. [Frontend Implementation](#3-frontend-implementation)
4. [Webhook Events](#4-webhook-events)
5. [Testing Checklist](#5-testing-checklist)
6. [Environment Variables](#6-environment-variables)

---

## 1. Stripe Dashboard Setup

### 1.1 Create Products and Prices

Navigate to **Stripe Dashboard → Products** and create:

| Product | Monthly Price | Annual Price | Price IDs (example) |
|---------|--------------|--------------|---------------------|
| **Pro** | $20/month | $200/year (~17% off) | `price_pro_monthly`, `price_pro_annual` |
| **Pro+** | $60/month | $600/year (~17% off) | `price_proplus_monthly`, `price_proplus_annual` |
| **Ultra** | $100/month | $1000/year (~17% off) | `price_ultra_monthly`, `price_ultra_annual` |

**Steps:**
1. Click "Add Product"
2. Name: "Pro" (or "Pro+", "Ultra")
3. Add two prices:
   - Monthly: Recurring, $20.00 USD, billed monthly
   - Annual: Recurring, $200.00 USD, billed yearly
4. Copy the Price IDs (e.g., `price_1ABC...`) for use in code

### 1.2 Configure Customer Portal

Navigate to **Stripe Dashboard → Settings → Billing → Customer portal**

Enable these features:
- [x] **Update payment methods** - Add/remove cards
- [x] **View invoices and billing history** - Download invoices
- [x] **Cancel subscriptions** - Allow cancellation (at period end)
- [x] **Switch plans/prices** - Allow upgrades/downgrades
  - Proration behavior: "Always invoice immediately" (recommended)
  - Products available: Select Pro, Pro+, Ultra

**Portal settings:**
- Business name: "Seer"
- Terms of service URL: `https://yourdomain.com/terms`
- Privacy policy URL: `https://yourdomain.com/privacy`

### 1.3 Set Up Webhook Endpoint

Navigate to **Stripe Dashboard → Developers → Webhooks**

1. Click "Add endpoint"
2. Endpoint URL: `https://api.yourdomain.com/api/v1/stripe/webhook`
3. Select events to listen to:
   - `checkout.session.completed`
   - `customer.subscription.created`
   - `customer.subscription.updated`
   - `customer.subscription.deleted`
   - `invoice.payment_succeeded`
   - `invoice.payment_failed`

4. Copy the **Webhook Signing Secret** (`whsec_...`)

---

## 2. Backend Implementation

### 2.1 Install Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
stripe = "^10.0.0"
```

Run:
```bash
uv add stripe
```

### 2.2 Configuration

Add to `shared/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_pro_monthly: str = ""
    stripe_price_pro_annual: str = ""
    stripe_price_proplus_monthly: str = ""
    stripe_price_proplus_annual: str = ""
    stripe_price_ultra_monthly: str = ""
    stripe_price_ultra_annual: str = ""

    # Frontend URL for redirects
    frontend_url: str = "http://localhost:5173"
```

### 2.3 Database Models

Create `shared/database/subscription_models.py`:

```python
from tortoise import fields, models
from enum import Enum


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    PRO_PLUS = "pro_plus"
    ULTRA = "ultra"


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    CANCELED = "canceled"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


class BillingProfileType(str, Enum):
    INDIVIDUAL = "individual"
    TEAM = "team"


class BillingProfile(models.Model):
    """Payer entity (individual user today, team later)."""

    id = fields.IntField(pk=True)
    type = fields.CharEnumField(BillingProfileType, default=BillingProfileType.INDIVIDUAL)
    owner_user = fields.ForeignKeyField("models.User", related_name="billing_profiles", on_delete=fields.CASCADE)
    stripe_customer_id = fields.CharField(max_length=255, unique=True, null=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "billing_profiles"


class BillingSubscription(models.Model):
    """Subscription record for a billing profile."""

    id = fields.IntField(pk=True)
    billing_profile = fields.OneToOneField("models.BillingProfile", related_name="subscription", on_delete=fields.CASCADE)

    stripe_subscription_id = fields.CharField(max_length=255, unique=True, null=True)
    tier = fields.CharEnumField(SubscriptionTier, default=SubscriptionTier.FREE)
    status = fields.CharEnumField(SubscriptionStatus, default=SubscriptionStatus.ACTIVE)

    current_period_start = fields.DatetimeField(null=True)
    current_period_end = fields.DatetimeField(null=True)
    cancel_at_period_end = fields.BooleanField(default=False)

    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta:
        table = "billing_subscriptions"
```

**Migration:**
```bash
aerich migrate --name add_user_subscriptions
aerich upgrade
```

### 2.4 Stripe Service

Create `api/subscriptions/stripe_service.py`:

```python
import stripe
from datetime import datetime
from typing import Optional
from shared.config import settings
from shared.database.models import User
from shared.database.subscription_models import UserSubscription, SubscriptionTier, SubscriptionStatus

# Initialize Stripe
stripe.api_key = settings.stripe_secret_key

# Price ID to tier mapping
PRICE_TO_TIER = {
    settings.stripe_price_pro_monthly: SubscriptionTier.PRO,
    settings.stripe_price_pro_annual: SubscriptionTier.PRO,
    settings.stripe_price_proplus_monthly: SubscriptionTier.PRO_PLUS,
    settings.stripe_price_proplus_annual: SubscriptionTier.PRO_PLUS,
    settings.stripe_price_ultra_monthly: SubscriptionTier.ULTRA,
    settings.stripe_price_ultra_annual: SubscriptionTier.ULTRA,
}


async def get_or_create_stripe_customer(user: User) -> str:
    """Get existing Stripe customer or create new one"""
    subscription = await UserSubscription.get_or_none(user=user)

    if subscription and subscription.stripe_customer_id:
        return subscription.stripe_customer_id

    # Create Stripe customer
    customer = stripe.Customer.create(
        email=user.email,
        name=f"{user.first_name or ''} {user.last_name or ''}".strip() or None,
        metadata={
            "user_id": user.user_id,  # Clerk user ID
            "seer_user_id": str(user.id),
        }
    )

    # Store customer ID
    if not subscription:
        subscription = await UserSubscription.create(user=user, stripe_customer_id=customer.id)
    else:
        subscription.stripe_customer_id = customer.id
        await subscription.save()

    return customer.id


async def create_checkout_session(
    user: User,
    price_id: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """Create Stripe Checkout session and return URL"""
    customer_id = await get_or_create_stripe_customer(user)

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        allow_promotion_codes=True,
        billing_address_collection="auto",
        metadata={
            "user_id": user.user_id,
        }
    )

    return session.url


async def create_portal_session(user: User, return_url: str) -> str:
    """Create Stripe Customer Portal session and return URL"""
    subscription = await UserSubscription.get_or_none(user=user)

    if not subscription or not subscription.stripe_customer_id:
        raise ValueError("User has no Stripe customer ID")

    session = stripe.billing_portal.Session.create(
        customer=subscription.stripe_customer_id,
        return_url=return_url,
    )

    return session.url


async def get_user_subscription(user: User) -> Optional[UserSubscription]:
    """Get user's subscription or create free tier"""
    subscription, _ = await UserSubscription.get_or_create(
        user=user,
        defaults={"tier": SubscriptionTier.FREE, "status": SubscriptionStatus.ACTIVE}
    )
    return subscription


async def sync_subscription_from_stripe(stripe_subscription: dict) -> None:
    """Sync subscription state from Stripe webhook data"""
    customer_id = stripe_subscription["customer"]
    subscription_id = stripe_subscription["id"]
    status = stripe_subscription["status"]

    # Find user by customer ID
    user_sub = await UserSubscription.get_or_none(stripe_customer_id=customer_id)
    if not user_sub:
        return  # Customer not in our system

    # Determine tier from price
    price_id = stripe_subscription["items"]["data"][0]["price"]["id"]
    tier = PRICE_TO_TIER.get(price_id, SubscriptionTier.FREE)

    # Map Stripe status to our status
    status_map = {
        "active": SubscriptionStatus.ACTIVE,
        "canceled": SubscriptionStatus.CANCELED,
        "past_due": SubscriptionStatus.PAST_DUE,
        "trialing": SubscriptionStatus.TRIALING,
        "incomplete": SubscriptionStatus.INCOMPLETE,
        "incomplete_expired": SubscriptionStatus.CANCELED,
        "unpaid": SubscriptionStatus.PAST_DUE,
    }

    # Update subscription
    user_sub.stripe_subscription_id = subscription_id
    user_sub.tier = tier
    user_sub.status = status_map.get(status, SubscriptionStatus.ACTIVE)
    user_sub.current_period_start = datetime.fromtimestamp(stripe_subscription["current_period_start"])
    user_sub.current_period_end = datetime.fromtimestamp(stripe_subscription["current_period_end"])
    user_sub.cancel_at_period_end = stripe_subscription.get("cancel_at_period_end", False)

    await user_sub.save()


async def handle_subscription_deleted(stripe_subscription: dict) -> None:
    """Handle subscription cancellation/deletion"""
    customer_id = stripe_subscription["customer"]

    user_sub = await UserSubscription.get_or_none(stripe_customer_id=customer_id)
    if not user_sub:
        return

    # Revert to free tier
    user_sub.tier = SubscriptionTier.FREE
    user_sub.status = SubscriptionStatus.ACTIVE
    user_sub.stripe_subscription_id = None
    user_sub.current_period_start = None
    user_sub.current_period_end = None
    user_sub.cancel_at_period_end = False

    await user_sub.save()
```

### 2.5 Clerk Metadata Sync

Create `api/subscriptions/clerk_sync.py`:

```python
import httpx
from shared.config import settings


async def sync_stripe_customer_to_clerk(clerk_user_id: str, stripe_customer_id: str) -> None:
    """
    Sync Stripe customer ID to Clerk user's public metadata.
    Requires CLERK_SECRET_KEY environment variable.
    """
    clerk_secret_key = getattr(settings, "clerk_secret_key", None)
    if not clerk_secret_key:
        return  # Skip if Clerk secret not configured

    async with httpx.AsyncClient() as client:
        await client.patch(
            f"https://api.clerk.com/v1/users/{clerk_user_id}",
            headers={
                "Authorization": f"Bearer {clerk_secret_key}",
                "Content-Type": "application/json",
            },
            json={
                "public_metadata": {
                    "stripe_customer_id": stripe_customer_id,
                }
            }
        )
```

Add to `shared/config.py`:
```python
clerk_secret_key: str = ""  # For updating Clerk user metadata
```

### 2.6 API Router

Create `api/subscriptions/router.py`:

```python
from fastapi import APIRouter, Request, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
import stripe

from shared.config import settings
from shared.database.models import User
from api.subscriptions.stripe_service import (
    create_checkout_session,
    create_portal_session,
    get_user_subscription,
    sync_subscription_from_stripe,
    handle_subscription_deleted,
    get_or_create_stripe_customer,
)
from api.subscriptions.clerk_sync import sync_stripe_customer_to_clerk

router = APIRouter(prefix="/subscriptions", tags=["subscriptions"])


def _require_user(request: Request) -> User:
    user = getattr(request.state, "db_user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# --- Request/Response Models ---

class CheckoutRequest(BaseModel):
    price_id: str


class CheckoutResponse(BaseModel):
    checkout_url: str


class PortalResponse(BaseModel):
    portal_url: str


class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False


class PricingResponse(BaseModel):
    prices: list[dict]


# --- Endpoints ---

@router.get("/pricing", response_model=PricingResponse)
async def get_pricing():
    """Get available subscription prices"""
    return PricingResponse(prices=[
        {
            "tier": "pro",
            "name": "Pro",
            "monthly": {"price": 20, "price_id": settings.stripe_price_pro_monthly},
            "annual": {"price": 200, "price_id": settings.stripe_price_pro_annual},
        },
        {
            "tier": "pro_plus",
            "name": "Pro+",
            "monthly": {"price": 60, "price_id": settings.stripe_price_proplus_monthly},
            "annual": {"price": 600, "price_id": settings.stripe_price_proplus_annual},
        },
        {
            "tier": "ultra",
            "name": "Ultra",
            "monthly": {"price": 100, "price_id": settings.stripe_price_ultra_monthly},
            "annual": {"price": 1000, "price_id": settings.stripe_price_ultra_annual},
        },
    ])


@router.get("/current", response_model=SubscriptionResponse)
async def get_current_subscription(request: Request):
    """Get current user's subscription status"""
    user = _require_user(request)
    subscription = await get_user_subscription(user)

    return SubscriptionResponse(
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_end=subscription.current_period_end.isoformat() if subscription.current_period_end else None,
        cancel_at_period_end=subscription.cancel_at_period_end,
    )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(request: Request, body: CheckoutRequest):
    """Create Stripe Checkout session for subscription"""
    user = _require_user(request)

    success_url = f"{settings.frontend_url}/settings/billing?success=true"
    cancel_url = f"{settings.frontend_url}/settings/billing?canceled=true"

    checkout_url = await create_checkout_session(
        user=user,
        price_id=body.price_id,
        success_url=success_url,
        cancel_url=cancel_url,
    )

    return CheckoutResponse(checkout_url=checkout_url)


@router.post("/portal", response_model=PortalResponse)
async def create_portal(request: Request):
    """Create Stripe Customer Portal session"""
    user = _require_user(request)

    return_url = f"{settings.frontend_url}/settings/billing"

    try:
        portal_url = await create_portal_session(user, return_url)
        return PortalResponse(portal_url=portal_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Webhook Handler ---

@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(alias="Stripe-Signature"),
):
    """Handle Stripe webhook events"""
    payload = await request.body()

    try:
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            settings.stripe_webhook_secret,
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        # Sync Stripe customer ID to Clerk
        customer_id = data.get("customer")
        user_id = data.get("metadata", {}).get("user_id")
        if customer_id and user_id:
            await sync_stripe_customer_to_clerk(user_id, customer_id)

    elif event_type in ("customer.subscription.created", "customer.subscription.updated"):
        await sync_subscription_from_stripe(data)

    elif event_type == "customer.subscription.deleted":
        await handle_subscription_deleted(data)

    elif event_type == "invoice.payment_failed":
        # Could send email notification here
        pass

    return {"status": "ok"}
```

### 2.7 Register Router

Update `api/router.py`:

```python
from api.subscriptions.router import router as subscriptions_router

# Add to router includes
api_v1_router.include_router(subscriptions_router)
```

### 2.8 Subscription Guard Middleware (Optional)

Create `api/subscriptions/middleware.py`:

```python
from functools import wraps
from fastapi import HTTPException, Request
from shared.database.subscription_models import SubscriptionTier, SubscriptionStatus


def require_subscription(min_tier: SubscriptionTier = SubscriptionTier.PRO):
    """Decorator to require minimum subscription tier"""
    tier_order = {
        SubscriptionTier.FREE: 0,
        SubscriptionTier.PRO: 1,
        SubscriptionTier.PRO_PLUS: 2,
        SubscriptionTier.ULTRA: 3,
    }

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, "db_user", None)
            if not user:
                raise HTTPException(status_code=401, detail="Authentication required")

            from api.subscriptions.stripe_service import get_user_subscription
            subscription = await get_user_subscription(user)

            if subscription.status != SubscriptionStatus.ACTIVE:
                raise HTTPException(status_code=403, detail="Subscription is not active")

            if tier_order[subscription.tier] < tier_order[min_tier]:
                raise HTTPException(
                    status_code=403,
                    detail=f"This feature requires {min_tier.value} subscription or higher"
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
```

**Usage example:**
```python
from api.subscriptions.middleware import require_subscription
from shared.database.subscription_models import SubscriptionTier

@router.post("/advanced-feature")
@require_subscription(min_tier=SubscriptionTier.PRO_PLUS)
async def advanced_feature(request: Request):
    # Only Pro+ and Ultra users can access this
    pass
```

---

## 3. Frontend Implementation

### 3.1 Install Dependencies

```bash
cd /home/lokesh/work/seer-frontend
npm install @stripe/stripe-js
```

### 3.2 API Client Methods

Add to `src/lib/api-client.ts` or create `src/lib/subscription-api.ts`:

```typescript
export interface SubscriptionInfo {
  tier: 'free' | 'pro' | 'pro_plus' | 'ultra';
  status: 'active' | 'canceled' | 'past_due' | 'trialing' | 'incomplete';
  current_period_end: string | null;
  cancel_at_period_end: boolean;
}

export interface PriceTier {
  tier: string;
  name: string;
  monthly: { price: number; price_id: string };
  annual: { price: number; price_id: string };
}

export interface PricingResponse {
  prices: PriceTier[];
}

export const subscriptionApi = {
  async getCurrentSubscription(): Promise<SubscriptionInfo> {
    const client = await getApiClient();
    return client.get('/api/v1/subscriptions/current');
  },

  async getPricing(): Promise<PricingResponse> {
    const client = await getApiClient();
    return client.get('/api/v1/subscriptions/pricing');
  },

  async createCheckout(priceId: string): Promise<{ checkout_url: string }> {
    const client = await getApiClient();
    return client.post('/api/v1/subscriptions/checkout', { price_id: priceId });
  },

  async createPortalSession(): Promise<{ portal_url: string }> {
    const client = await getApiClient();
    return client.post('/api/v1/subscriptions/portal', {});
  },
};
```

### 3.3 Subscription Store

Create `src/stores/subscriptionStore.ts`:

```typescript
import { create } from 'zustand';
import { subscriptionApi, SubscriptionInfo, PriceTier } from '@/lib/subscription-api';

interface SubscriptionState {
  subscription: SubscriptionInfo | null;
  pricing: PriceTier[] | null;
  isLoading: boolean;
  error: string | null;

  fetchSubscription: () => Promise<void>;
  fetchPricing: () => Promise<void>;
  redirectToCheckout: (priceId: string) => Promise<void>;
  redirectToPortal: () => Promise<void>;
}

export const useSubscriptionStore = create<SubscriptionState>((set) => ({
  subscription: null,
  pricing: null,
  isLoading: false,
  error: null,

  fetchSubscription: async () => {
    set({ isLoading: true, error: null });
    try {
      const subscription = await subscriptionApi.getCurrentSubscription();
      set({ subscription, isLoading: false });
    } catch (err) {
      set({ error: 'Failed to fetch subscription', isLoading: false });
    }
  },

  fetchPricing: async () => {
    try {
      const { prices } = await subscriptionApi.getPricing();
      set({ pricing: prices });
    } catch (err) {
      set({ error: 'Failed to fetch pricing' });
    }
  },

  redirectToCheckout: async (priceId: string) => {
    set({ isLoading: true, error: null });
    try {
      const { checkout_url } = await subscriptionApi.createCheckout(priceId);
      window.location.href = checkout_url;
    } catch (err) {
      set({ error: 'Failed to create checkout session', isLoading: false });
    }
  },

  redirectToPortal: async () => {
    set({ isLoading: true, error: null });
    try {
      const { portal_url } = await subscriptionApi.createPortalSession();
      window.location.href = portal_url;
    } catch (err) {
      set({ error: 'Failed to open billing portal', isLoading: false });
    }
  },
}));
```

### 3.4 Billing Settings Page

Create `src/pages/settings/BillingSettings.tsx`:

```tsx
import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { useSubscriptionStore } from '@/stores/subscriptionStore';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Check, Loader2 } from 'lucide-react';

const TIER_FEATURES: Record<string, string[]> = {
  free: ['Basic workflows', 'Limited runs/month', 'Community support'],
  pro: ['More workflow runs', 'Priority execution', 'Email support'],
  pro_plus: ['High volume runs', 'Advanced features', 'Priority support'],
  ultra: ['Unlimited runs', 'All features', 'Dedicated support'],
};

export function BillingSettings() {
  const [searchParams] = useSearchParams();
  const {
    subscription,
    pricing,
    isLoading,
    fetchSubscription,
    fetchPricing,
    redirectToCheckout,
    redirectToPortal,
  } = useSubscriptionStore();

  useEffect(() => {
    fetchSubscription();
    fetchPricing();
  }, []);

  const success = searchParams.get('success') === 'true';
  const canceled = searchParams.get('canceled') === 'true';

  if (isLoading && !subscription) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Billing</h2>
        <p className="text-muted-foreground">
          Manage your subscription and billing settings.
        </p>
      </div>

      {success && (
        <div className="rounded-lg bg-green-50 p-4 text-green-800 dark:bg-green-900/20 dark:text-green-400">
          Successfully subscribed! Your plan is now active.
        </div>
      )}

      {canceled && (
        <div className="rounded-lg bg-yellow-50 p-4 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400">
          Checkout was canceled. No charges were made.
        </div>
      )}

      {/* Current Plan */}
      <Card>
        <CardHeader>
          <CardTitle>Current Plan</CardTitle>
          <CardDescription>Your current subscription status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-2xl font-semibold capitalize">
                {subscription?.tier.replace('_', ' ') || 'Free'}
              </p>
              <p className="text-sm text-muted-foreground">
                Status: <Badge variant={subscription?.status === 'active' ? 'default' : 'secondary'}>
                  {subscription?.status || 'Active'}
                </Badge>
              </p>
              {subscription?.current_period_end && (
                <p className="text-sm text-muted-foreground mt-1">
                  {subscription.cancel_at_period_end
                    ? `Cancels on ${new Date(subscription.current_period_end).toLocaleDateString()}`
                    : `Renews on ${new Date(subscription.current_period_end).toLocaleDateString()}`}
                </p>
              )}
            </div>
            {subscription?.tier !== 'free' && (
              <Button variant="outline" onClick={redirectToPortal} disabled={isLoading}>
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Manage Subscription
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Pricing Plans */}
      <div>
        <h3 className="text-lg font-semibold mb-4">
          {subscription?.tier === 'free' ? 'Upgrade Your Plan' : 'Available Plans'}
        </h3>
        <div className="grid gap-4 md:grid-cols-3">
          {pricing?.map((plan) => (
            <Card key={plan.tier} className={subscription?.tier === plan.tier ? 'border-primary' : ''}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  {plan.name}
                  {subscription?.tier === plan.tier && <Badge>Current</Badge>}
                </CardTitle>
                <CardDescription>
                  <span className="text-2xl font-bold">${plan.monthly.price}</span>/month
                  <br />
                  <span className="text-sm">or ${plan.annual.price}/year (save ~17%)</span>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 mb-4">
                  {TIER_FEATURES[plan.tier]?.map((feature) => (
                    <li key={feature} className="flex items-center text-sm">
                      <Check className="h-4 w-4 mr-2 text-green-500" />
                      {feature}
                    </li>
                  ))}
                </ul>
                {subscription?.tier !== plan.tier && (
                  <div className="space-y-2">
                    <Button
                      className="w-full"
                      onClick={() => redirectToCheckout(plan.monthly.price_id)}
                      disabled={isLoading}
                    >
                      {isLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                      Subscribe Monthly
                    </Button>
                    <Button
                      variant="outline"
                      className="w-full"
                      onClick={() => redirectToCheckout(plan.annual.price_id)}
                      disabled={isLoading}
                    >
                      Subscribe Annually
                    </Button>
                  </div>
                )}
                {subscription?.tier === plan.tier && subscription.tier !== 'free' && (
                  <Button variant="outline" className="w-full" onClick={redirectToPortal}>
                    Change Plan
                  </Button>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
```

### 3.5 Add Route

Update your router configuration to add the billing page:

```tsx
// In your routes configuration
import { BillingSettings } from '@/pages/settings/BillingSettings';

// Add route
<Route path="/settings/billing" element={<BillingSettings />} />
```

### 3.6 Navigation Link

Add a link to billing in your settings navigation:

```tsx
<Link to="/settings/billing">Billing</Link>
```

---

## 4. Webhook Events

### Events to Handle

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Sync customer ID to Clerk, log successful checkout |
| `customer.subscription.created` | Create/update subscription record |
| `customer.subscription.updated` | Update tier, status, period dates |
| `customer.subscription.deleted` | Revert to free tier |
| `invoice.payment_failed` | Update status to past_due, notify user |
| `invoice.payment_succeeded` | Clear any past_due status |

### Local Testing with Stripe CLI

```bash
# Install Stripe CLI
brew install stripe/stripe-cli/stripe

# Login
stripe login

# Forward webhooks to local server
stripe listen --forward-to localhost:8000/api/v1/subscriptions/webhook

# The CLI will print a webhook signing secret - use this for local testing
# e.g., whsec_...
```

---

## 5. Testing Checklist

### Stripe Dashboard Setup
- [ ] Create Pro product with monthly ($20) and annual ($200) prices
- [ ] Create Pro+ product with monthly ($60) and annual ($600) prices
- [ ] Create Ultra product with monthly ($100) and annual ($1000) prices
- [ ] Configure Customer Portal (payments, invoices, cancellation, plan switching)
- [ ] Create webhook endpoint pointing to your API
- [ ] Note all price IDs and webhook signing secret

### Backend Testing
- [ ] Run database migration for `user_subscriptions` table
- [ ] Test `/api/v1/subscriptions/pricing` returns all plans
- [ ] Test `/api/v1/subscriptions/current` returns free tier for new users
- [ ] Test `/api/v1/subscriptions/checkout` creates valid Stripe session
- [ ] Test `/api/v1/subscriptions/portal` creates valid portal session
- [ ] Test webhook signature validation
- [ ] Test `checkout.session.completed` webhook syncs customer ID
- [ ] Test `customer.subscription.created` webhook creates subscription record
- [ ] Test `customer.subscription.updated` webhook updates tier/status
- [ ] Test `customer.subscription.deleted` webhook reverts to free

### Frontend Testing
- [ ] Billing page loads and shows current subscription (free)
- [ ] Pricing cards display correctly with monthly/annual options
- [ ] "Subscribe Monthly" redirects to Stripe Checkout
- [ ] "Subscribe Annually" redirects to Stripe Checkout
- [ ] Success redirect shows success message
- [ ] Cancel redirect shows canceled message
- [ ] "Manage Subscription" opens Stripe Customer Portal
- [ ] After subscribing, current plan shows correctly

### End-to-End Flows
- [ ] New user → Free tier → Subscribe to Pro (monthly) → Subscription active
- [ ] Pro user → Upgrade to Ultra via Portal → Subscription updated
- [ ] Pro user → Downgrade to Pro+ via Portal → Subscription updated
- [ ] User → Cancel subscription in Portal → Shows "Cancels on [date]"
- [ ] User → Update payment method in Portal → Card updated
- [ ] User → Download invoice in Portal → PDF downloads

### Test Cards (Stripe Test Mode)
- Success: `4242 4242 4242 4242`
- Decline: `4000 0000 0000 0002`
- Requires auth: `4000 0025 0000 3155`

---

## 6. Environment Variables

### Backend (`.env`)

```env
# Stripe
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PRO_MONTHLY=price_...
STRIPE_PRICE_PRO_ANNUAL=price_...
STRIPE_PRICE_PROPLUS_MONTHLY=price_...
STRIPE_PRICE_PROPLUS_ANNUAL=price_...
STRIPE_PRICE_ULTRA_MONTHLY=price_...
STRIPE_PRICE_ULTRA_ANNUAL=price_...

# Clerk (for metadata sync)
CLERK_SECRET_KEY=sk_...

# Frontend URL for redirects
FRONTEND_URL=http://localhost:5173
```

### Frontend (`.env`)

```env
# No Stripe keys needed on frontend - we use backend to create sessions
VITE_BACKEND_API_URL=http://localhost:8000
```

### Production

```env
# Use live keys
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...  # Get from production webhook in Stripe
STRIPE_PRICE_PRO_MONTHLY=price_...  # Live price IDs
# ... etc
```

---

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend      │────▶│     Stripe      │
│   (React/TS)    │     │    (FastAPI)     │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                       │
        │  1. Subscribe click    │                       │
        │───────────────────────▶│                       │
        │                        │  2. Create checkout   │
        │                        │──────────────────────▶│
        │                        │  3. Return session URL│
        │  4. Redirect           │◀──────────────────────│
        │◀───────────────────────│                       │
        │                        │                       │
        │  5. User completes payment on Stripe           │
        │────────────────────────────────────────────────▶
        │                        │                       │
        │                        │  6. Webhook event     │
        │                        │◀──────────────────────│
        │                        │  7. Update DB         │
        │                        │  8. Sync to Clerk     │
        │                        │                       │
        │  9. Redirect back      │                       │
        │◀───────────────────────────────────────────────│
        │                        │                       │
        │ 10. Fetch subscription │                       │
        │───────────────────────▶│                       │
```

---

## Implementation Order

### Phase 1: Backend Foundation (Backend Developer)
1. Add Stripe dependency
2. Add config settings
3. Create database model and migration
4. Implement Stripe service
5. Implement API router
6. Register router

### Phase 2: Stripe Setup (Admin)
1. Create products and prices in Stripe Dashboard
2. Configure Customer Portal
3. Create webhook endpoint
4. Get all price IDs and secrets

### Phase 3: Frontend Integration (Frontend Developer)
1. Create subscription API client
2. Create Zustand store
3. Build billing settings page
4. Add routes and navigation

### Phase 4: Testing (Both)
1. Test with Stripe CLI locally
2. End-to-end flow testing
3. Deploy to staging
4. Test production webhooks

---

## Notes

- **Annual discount**: 2 months free (~17% discount) is standard. Adjust pricing if needed.
- **Proration**: Stripe automatically prorates when switching plans mid-cycle.
- **Cancellation**: Users cancel at period end (not immediate) to avoid refunds.
- **Usage limits**: Add to `UserSubscription` model and check in relevant endpoints when ready.
- **Email notifications**: Consider adding email on payment failure using a service like Resend.
