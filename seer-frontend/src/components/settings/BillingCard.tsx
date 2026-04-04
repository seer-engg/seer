
import { useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Link } from "react-router-dom";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import {
  CreditCard,
  Loader2,
  Users,
  Info,
} from "lucide-react";
import { useSubscriptionStore } from "@/stores/subscriptionStore";
import { useOrganizationStore } from "@/stores/organizationStore";
import { canManageBilling } from "@/types/organization";
import type { SubscriptionTier } from "@/lib/subscription-api";

export function BillingCard() {
  const subscription = useSubscriptionStore((state) => state.subscription);
  const isLoadingSubscription = useSubscriptionStore((state) => state.isLoading);
  const fetchSubscription = useSubscriptionStore((state) => state.fetchSubscription);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const getCurrentRole = useOrganizationStore((s) => s.getCurrentRole);

  const role = getCurrentRole();
  const isTeamOrg = currentOrganization?.type === 'team';
  const canManage = canManageBilling(role);

  useEffect(() => {
    if (!subscription && !isLoadingSubscription) {
      fetchSubscription();
    }
  }, [subscription, isLoadingSubscription, fetchSubscription]);

  const tierLabel: Record<SubscriptionTier, string> = {
    free: "Free",
    pro: "Pro",
    pro_plus: "Pro+",
    ultra: "Ultra",
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <CreditCard className="h-5 w-5 text-seer" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">
                {isTeamOrg ? 'Team Plan' : 'Billing & Plans'}
              </CardTitle>
              {isTeamOrg && (
                <Badge variant="outline" className="h-5 px-1.5 text-[10px] bg-seer/10 text-seer border-seer/20">
                  <Users className="h-3 w-3 mr-1" />
                  Shared
                </Badge>
              )}
            </div>
            <CardDescription>
              {isTeamOrg
                ? `Shared by ${currentOrganization?.memberCount ?? 0} team members`
                : 'Manage your subscription and payment details'
              }
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="rounded-lg border bg-secondary/40 p-3">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-[0.08em] text-muted-foreground">Current plan</p>
              <p className="text-sm font-semibold">
                {isLoadingSubscription && !subscription ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-seer" />
                    Loading...
                  </span>
                ) : (
                  tierLabel[subscription?.tier ?? "free"]
                )}
              </p>
            </div>
            <Badge variant="outline" className="bg-background">
              {subscription?.status ?? "active"}
            </Badge>
          </div>
        </div>

        {canManage ? (
          <Button asChild variant="brand" className="w-full sm:w-auto">
            <Link to="/settings/billing">Manage Subscription</Link>
          </Button>
        ) : isTeamOrg ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="inline-flex items-center gap-1.5 cursor-help">
                    <Info className="h-4 w-4" />
                    <span>Contact team owner to manage billing</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Only the team owner can manage billing settings</p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
        ) : (
          <Button asChild variant="brand" className="w-full sm:w-auto">
            <Link to="/settings/billing">Manage Subscription</Link>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
