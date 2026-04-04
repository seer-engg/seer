import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CreditCard, Sparkles } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { useUIStore } from '@/stores/uiStore';
import { useSubscriptionStore } from '@/stores/subscriptionStore';
import { BackendAPIError } from '@/lib/api-client';
import type { PriceTier, SubscriptionInfo } from '@/lib/subscription-api';

const DEFAULT_BENEFITS = ['More workflow runs', 'Priority execution', 'Advanced features'];

function getNextPlanBenefits(nextPlan: PriceTier | undefined): string[] {
  if (nextPlan?.upgrade_benefits?.length) return nextPlan.upgrade_benefits;
  if (nextPlan?.features?.length) return nextPlan.features.slice(0, 3);
  return DEFAULT_BENEFITS;
}

function getUpgradeInfo(subscription: SubscriptionInfo | null, pricing: PriceTier[] | null) {
  const currentTier = subscription?.tier || 'free';
  const currentPlan = pricing?.find((p) => p.tier === currentTier);
  const tierLabel = currentPlan?.name ?? (currentTier === 'free' ? 'Free' : currentTier);

  const sortedPlans = [...(pricing ?? [])].sort((a, b) => (a.sort_order ?? 0) - (b.sort_order ?? 0));
  const currentIndex = sortedPlans.findIndex((p) => p.tier === currentTier);
  const nextPlan = currentIndex >= 0 && currentIndex < sortedPlans.length - 1
    ? sortedPlans[currentIndex + 1]
    : sortedPlans[0];

  return { tierLabel, benefits: getNextPlanBenefits(nextPlan) };
}

/**
 * PaymentRequiredModal - Shown when user encounters a 402 error
 * Listens globally for 402 API errors and provides a clear upgrade path
 */
export function PaymentRequiredModal() {
  const navigate = useNavigate();
  const isOpen = useUIStore((state) => state.isPaymentModalOpen);
  const setOpen = useUIStore((state) => state.setPaymentModalOpen);
  const subscription = useSubscriptionStore((state) => state.subscription);
  const pricing = useSubscriptionStore((state) => state.pricing);

  useEffect(() => {
    const handleApiError = (event: Event) => {
      const customEvent = event as CustomEvent<BackendAPIError>;
      if (customEvent.detail?.status === 402) {
        setOpen(true);
      }
    };

    window.addEventListener('api-error', handleApiError);
    return () => window.removeEventListener('api-error', handleApiError);
  }, [setOpen]);

  const handleUpgrade = () => {
    setOpen(false);
    navigate('/settings/billing');
  };

  const handleClose = () => {
    setOpen(false);
  };

  const { tierLabel, benefits } = getUpgradeInfo(subscription, pricing);

  return (
    <Dialog open={isOpen} onOpenChange={setOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <div className="rounded-full bg-warning/10 p-2">
              <CreditCard className="h-5 w-5 text-warning" />
            </div>
            <DialogTitle>Upgrade Required</DialogTitle>
          </div>
          <DialogDescription className="pt-2">
            You've reached the limit of your {tierLabel} plan. Upgrade to continue using this feature.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 py-4">
          <div className="rounded-lg border bg-muted/50 p-4">
            <p className="text-sm text-muted-foreground">
              Upgrade your plan to unlock:
            </p>
            <ul className="mt-2 space-y-1.5">
              {benefits.map((benefit) => (
                <li key={benefit} className="flex items-center gap-2 text-sm">
                  <Sparkles className="h-3.5 w-3.5 text-primary" />
                  <span>{benefit}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          <Button onClick={handleUpgrade}>
            <CreditCard className="mr-2 h-4 w-4" />
            Upgrade Plan
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
