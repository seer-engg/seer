import { useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { BillingHistorySection } from "@/components/settings/BillingHistorySection";
import { OverageSettingsCard } from "@/components/settings/OverageSettingsCard";
import { cn } from "@/lib/utils";
import { useSubscriptionStore } from "@/stores/subscriptionStore";
import { useOrganizationStore } from "@/stores/organizationStore";
import { canManageBilling } from "@/types/organization";
import type { PriceTier, SubscriptionInfo, SubscriptionTier } from "@/lib/subscription-api";
import { ArrowLeft, Check, CreditCard, Loader2, Sparkles, ShieldAlert, Users } from "lucide-react";

const formatDate = (isoDate?: string | null): string | null => {
  if (!isoDate) return null;
  const date = new Date(isoDate);
  return Number.isNaN(date.getTime()) ? null : date.toLocaleDateString();
};

function StatusMessages({
  success,
  canceled,
  error,
}: {
  success: boolean;
  canceled: boolean;
  error: string | null;
}) {
  if (!success && !canceled && !error) {
    return null;
  }

  return (
    <div className="space-y-3">
      {success && (
        <Alert className="border-success/50 bg-success/5 text-success-foreground">
          <Sparkles className="h-4 w-4" />
          <AlertTitle>Subscription activated</AlertTitle>
          <AlertDescription>Your plan is live. Billing details are synced from Stripe.</AlertDescription>
        </Alert>
      )}
      {canceled && (
        <Alert className="border-warning/50 bg-warning/10 text-warning-foreground">
          <AlertTitle>Checkout canceled</AlertTitle>
          <AlertDescription>No charges were made. You can restart checkout anytime.</AlertDescription>
        </Alert>
      )}
      {error && (
        <Alert variant="destructive">
          <AlertTitle>Billing error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
    </div>
  );
}

interface CurrentPlanCardProps {
  subscription: SubscriptionInfo | null;
  pricing: PriceTier[] | null;
  isLoading: boolean;
  onPortal: () => void;
  onCheckout: (tier: string, interval: string) => void;
}

interface CurrentPlanActionProps {
  canManageInPortal: boolean;
  isLoading: boolean;
  defaultTier?: SubscriptionTier;
  onPortal: () => void;
  onCheckout: (tier: string, interval: string) => void;
}

function CurrentPlanAction({
  canManageInPortal,
  isLoading,
  defaultTier,
  onPortal,
  onCheckout,
}: CurrentPlanActionProps) {
  if (canManageInPortal) {
    return (
      <Button variant="outline" onClick={onPortal} disabled={isLoading}>
        {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <CreditCard className="h-4 w-4" />}
        Open billing portal
      </Button>
    );
  }

  return (
    <Button
      variant="brand"
      onClick={() => defaultTier && onCheckout(defaultTier, "month")}
      disabled={isLoading || !defaultTier}
    >
      Start a paid plan
    </Button>
  );
}

function getPlanDisplayInfo(subscription: SubscriptionInfo | null, pricing: PriceTier[] | null) {
  const tier = subscription?.tier ?? "free";
  const status = subscription?.status ?? "active";
  const nextRenewal = formatDate(subscription?.current_period_end);
  const defaultTier = pricing?.[0]?.tier;
  const canManageInPortal = tier !== "free";
  const currentPlan = pricing?.find((p) => p.tier === tier);
  const displayName = currentPlan?.name ?? (tier === "free" ? "Free" : tier);
  const features = currentPlan?.features ?? [];
  return { tier, status, nextRenewal, defaultTier, canManageInPortal, displayName, features };
}

function CurrentPlanCard({ subscription, pricing, isLoading, onPortal, onCheckout }: CurrentPlanCardProps) {
  const { status, nextRenewal, defaultTier, canManageInPortal, displayName, features } =
    getPlanDisplayInfo(subscription, pricing);

  return (
    <Card className="border-seer/20 bg-gradient-to-r from-seer/10 via-card to-background">
      <CardHeader className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-base">Current plan</CardTitle>
          <CardDescription>Your subscription status from Stripe</CardDescription>
        </div>
        <Badge variant={status === "active" ? "default" : "secondary"}>{status}</Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-2">
          <p className="text-2xl font-semibold">{displayName}</p>
          {nextRenewal && (
            <p className="text-sm text-muted-foreground">
              {subscription?.cancel_at_period_end ? "Cancels on " : "Renews on "}
              {nextRenewal}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            {features.map((feature) => (
              <Badge key={feature} variant="outline" className="bg-secondary/40">
                <Check className="mr-1.5 h-3.5 w-3.5 text-success" />
                {feature}
              </Badge>
            ))}
          </div>
        </div>
        <CurrentPlanAction
          canManageInPortal={canManageInPortal}
          isLoading={isLoading}
          defaultTier={defaultTier}
          onPortal={onPortal}
          onCheckout={onCheckout}
        />
      </CardContent>
    </Card>
  );
}

interface PlanCardProps {
  plan: PriceTier;
  isCurrent: boolean;
  isLoading: boolean;
  onPortal: () => void;
  onCheckout: (tier: string, interval: string) => void;
}

const formatPrice = (price: number) => {
  return (price / 100).toFixed(2);
};
function PlanCard({ plan, isCurrent, isLoading, onPortal, onCheckout }: PlanCardProps) {
  const monthlyAvailable = Boolean(plan.monthly.price_id);
  const annualAvailable = Boolean(plan.annual.price_id);

  const actualMonthlyPrice = formatPrice(plan.monthly.price);
  const actualAnnualPrice = formatPrice(plan.annual.price);
  const hasOriginalMonthly = plan.monthly.original_price != null;
  const hasOriginalAnnual = plan.annual.original_price != null;
  const features = plan.features ?? [];

  // Compute savings percentage dynamically from monthly vs annual
  const monthlyCostPerYear = plan.monthly.price * 12;
  const savingsPercent = monthlyCostPerYear > 0
    ? Math.round(((monthlyCostPerYear - plan.annual.price) / monthlyCostPerYear) * 100)
    : 0;

  return (
    <Card
      className={cn(
        "relative overflow-hidden border-muted",
        isCurrent && "border-seer shadow-[0_10px_40px_-20px_rgba(131,56,236,0.7)]",
      )}
    >
      <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-seer via-purple-500 to-seer opacity-70" />
      <CardHeader className="space-y-3 pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">{plan.name}</CardTitle>
          {isCurrent ? <Badge>Current</Badge> : null}
        </div>
        <CardDescription>
          <div className="space-y-1">
            <div className="flex items-baseline gap-2">
              {hasOriginalMonthly && (
                <span className="text-lg font-medium text-muted-foreground line-through">
                  ${formatPrice(plan.monthly.original_price!)}
                </span>
              )}
              <span className="text-3xl font-semibold">${actualMonthlyPrice}</span>
              <span className="text-sm text-muted-foreground"> / month</span>
            </div>
            <div className="flex items-baseline gap-2 text-xs text-muted-foreground">
              {hasOriginalAnnual && (
                <span className="line-through">${formatPrice(plan.annual.original_price!)}</span>
              )}
              <span className="font-medium">${actualAnnualPrice} per year</span>
              {savingsPercent > 0 && (
                <span>(save ~{savingsPercent}%)</span>
              )}
            </div>
            {plan.badge && (
              <Badge variant="outline" className="mt-1 bg-seer/10 text-seer border-seer/20 text-xs">
                {plan.badge}
              </Badge>
            )}
          </div>
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <ul className="space-y-2 text-sm text-muted-foreground">
          {features.map((feature) => (
            <li key={feature} className="flex items-center gap-2">
              <Check className="h-4 w-4 text-success" />
              <span className="text-foreground">{feature}</span>
            </li>
          ))}
        </ul>

        {isCurrent ? (
          <Button variant="outline" className="w-full" onClick={onPortal} disabled={isLoading}>
            Manage plan
          </Button>
        ) : (
          <div className="space-y-2">
            <Button
              className="w-full"
              variant="brand"
              onClick={() => monthlyAvailable && onCheckout(plan.tier, "month")}
              disabled={isLoading || !monthlyAvailable}
            >
              {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Subscribe monthly
            </Button>
            <Button
              className="w-full"
              variant="outline"
              onClick={() => annualAvailable && onCheckout(plan.tier, "year")}
              disabled={isLoading || !annualAvailable}
            >
              Subscribe annually
            </Button>
            {(!monthlyAvailable || !annualAvailable) && (
              <p className="text-xs text-warning-foreground/80">
                Stripe price IDs are missing for this tier.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

interface PricingGridProps {
  pricing: PriceTier[] | null;
  subscription: SubscriptionInfo | null;
  isLoading: boolean;
  onPortal: () => void;
  onCheckout: (tier: string, interval: string) => void;
}

function PricingGrid({ pricing, subscription, isLoading, onPortal, onCheckout }: PricingGridProps) {
  if (!pricing?.length) {
    return (
      <Card className="md:col-span-3">
        <CardContent className="flex items-center gap-3 py-8 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading pricing from Stripe...
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      {pricing.map((plan) => (
        <PlanCard
          key={plan.tier}
          plan={plan}
          isCurrent={subscription?.tier === plan.tier}
          isLoading={isLoading}
          onPortal={onPortal}
          onCheckout={onCheckout}
        />
      ))}
    </div>
  );
}

/* eslint-disable max-lines-per-function, complexity */
export function BillingSettings() {
  const [searchParams] = useSearchParams();
  const subscription = useSubscriptionStore((state) => state.subscription);
  const pricing = useSubscriptionStore((state) => state.pricing);
  const isLoading = useSubscriptionStore((state) => state.isLoading);
  const error = useSubscriptionStore((state) => state.error);
  const fetchSubscription = useSubscriptionStore((state) => state.fetchSubscription);
  const fetchPricing = useSubscriptionStore((state) => state.fetchPricing);
  const redirectToCheckout = useSubscriptionStore((state) => state.redirectToCheckout);
  const redirectToPortal = useSubscriptionStore((state) => state.redirectToPortal);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const getCurrentRole = useOrganizationStore((s) => s.getCurrentRole);
  const role = getCurrentRole();
  const isTeamOrg = currentOrganization?.type === 'team';
  const canManage = canManageBilling(role);

  useEffect(() => {
    fetchSubscription();
    fetchPricing();
  }, [fetchSubscription, fetchPricing]);

  const success = searchParams.get("success") === "true";
  const canceled = searchParams.get("canceled") === "true";

  // Non-owners in team context cannot access billing
  if (isTeamOrg && !canManage) {
    return (
      <div className="h-full overflow-y-auto scrollbar-thin">
        <div className="mx-auto max-w-6xl px-6 py-8 space-y-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Billing</p>
              <h1 className="text-2xl font-semibold text-gradient-seer">Access Restricted</h1>
            </div>
            <Button asChild variant="outline" size="sm" className="border-dashed">
              <Link to="/settings">
                <ArrowLeft className="h-4 w-4" />
                Back to settings
              </Link>
            </Button>
          </div>

          <Card className="border-amber-500/30 bg-amber-500/5">
            <CardContent className="flex items-center gap-4 py-8">
              <div className="h-12 w-12 rounded-full bg-amber-500/10 flex items-center justify-center">
                <ShieldAlert className="h-6 w-6 text-amber-500" />
              </div>
              <div>
                <h3 className="font-medium text-lg">Billing Access Required</h3>
                <p className="text-muted-foreground text-sm">
                  Only the team owner can manage billing settings for <strong>{currentOrganization?.name}</strong>.
                  Please contact your team owner to make changes to the subscription.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (isLoading && !subscription) {
    return (
      <div className="flex h-full items-center justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-seer" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto scrollbar-thin">
      <div className="mx-auto max-w-6xl px-6 py-8 space-y-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <p className="text-xs uppercase tracking-[0.25em] text-muted-foreground">Billing</p>
              {isTeamOrg && (
                <Badge variant="outline" className="h-5 px-1.5 text-[10px] bg-seer/10 text-seer border-seer/20">
                  <Users className="h-3 w-3 mr-1" />
                  {currentOrganization?.name}
                </Badge>
              )}
            </div>
            <h1 className="text-2xl font-semibold text-gradient-seer">
              {isTeamOrg ? 'Team Subscription' : 'Subscriptions'}
            </h1>
          </div>
          <div className="flex flex-wrap gap-3">
            <Button asChild variant="outline" size="sm" className="border-dashed">
              <Link to="/settings">
                <ArrowLeft className="h-4 w-4" />
                Back to settings
              </Link>
            </Button>
            {subscription?.tier && subscription.tier !== "free" ? (
              <Button variant="brand" size="sm" onClick={redirectToPortal} disabled={isLoading}>
                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <CreditCard className="h-4 w-4" />}
                Manage in Stripe
              </Button>
            ) : null}
          </div>
        </div>

        <StatusMessages success={success} canceled={canceled} error={error} />

        <CurrentPlanCard
          subscription={subscription}
          pricing={pricing}
          isLoading={isLoading}
          onPortal={redirectToPortal}
          onCheckout={redirectToCheckout}
        />

        <div className="space-y-3">
          <div>
            <h2 className="text-lg font-semibold">
              {isTeamOrg ? 'Upgrade your team' : 'Upgrade your workspace'}
            </h2>
            <p className="text-sm text-muted-foreground">
              {isTeamOrg
                ? `Choose a plan for your team of ${currentOrganization?.memberCount ?? 0} members. All members share the quota.`
                : 'Choose monthly or annual billing.'
              }
            </p>
          </div>
          <PricingGrid
            pricing={pricing}
            subscription={subscription}
            isLoading={isLoading}
            onPortal={redirectToPortal}
            onCheckout={redirectToCheckout}
          />
        </div>

        {/* Overage Settings for paid tier users */}
        <OverageSettingsCard />

        <BillingHistorySection />
      </div>
    </div>
  );
}
