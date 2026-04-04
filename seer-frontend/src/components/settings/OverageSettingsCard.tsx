import { useEffect, useState, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertTriangle,
  CheckCircle2,
  DollarSign,
  Loader2,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { useOverageStore } from "@/stores/overageStore";
import { useSubscriptionStore } from "@/stores/subscriptionStore";
import { cn } from "@/lib/utils";

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Helpers
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_CAP_CENTS = 5000;
const DEFAULT_MIN_CAP_CENTS = 500;
const DEFAULT_MAX_CAP_CENTS = 100000;
const DEFAULT_WARNING_THRESHOLD = 0.8;

// ─────────────────────────────────────────────────────────────────────────────
// Custom Hook
// ─────────────────────────────────────────────────────────────────────────────

function useOverageSettings() {
  const subscription = useSubscriptionStore((state) => state.subscription);
  const config = useOverageStore((state) => state.config);
  const settings = useOverageStore((state) => state.settings);
  const isLoading = useOverageStore((state) => state.isLoading);
  const isEnabling = useOverageStore((state) => state.isEnabling);
  const isDisabling = useOverageStore((state) => state.isDisabling);
  const isUpdatingCap = useOverageStore((state) => state.isUpdatingCap);
  const error = useOverageStore((state) => state.error);
  const fetchConfig = useOverageStore((state) => state.fetchConfig);
  const fetchSettings = useOverageStore((state) => state.fetchSettings);
  const enableOverage = useOverageStore((state) => state.enableOverage);
  const disableOverage = useOverageStore((state) => state.disableOverage);
  const updateSpendingCap = useOverageStore((state) => state.updateSpendingCap);
  const clearError = useOverageStore((state) => state.clearError);

  const [pendingCapCents, setPendingCapCents] = useState<number | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig();
    fetchSettings();
  }, [fetchConfig, fetchSettings]);

  useEffect(() => {
    if (settings?.spending_cap_cents && pendingCapCents === null) {
      setPendingCapCents(settings.spending_cap_cents);
    }
  }, [settings?.spending_cap_cents, pendingCapCents]);

  useEffect(() => {
    if (!successMessage) return;
    const timer = setTimeout(() => setSuccessMessage(null), 5000);
    return () => clearTimeout(timer);
  }, [successMessage]);

  const handleToggleOverage = useCallback(async () => {
    clearError();
    setSuccessMessage(null);

    if (settings?.enabled) {
      const result = await disableOverage();
      if (result.success) setSuccessMessage(result.message);
    } else {
      const capCents = pendingCapCents ?? config?.default_cap_cents ?? DEFAULT_CAP_CENTS;
      const success = await enableOverage(capCents);
      if (success) setSuccessMessage("Usage-based pricing has been enabled.");
    }
  }, [clearError, settings?.enabled, disableOverage, pendingCapCents, config?.default_cap_cents, enableOverage]);

  const handleUpdateCap = useCallback(async () => {
    if (pendingCapCents === null) return;
    if (pendingCapCents === settings?.spending_cap_cents) return;

    clearError();
    setSuccessMessage(null);

    const success = await updateSpendingCap(pendingCapCents);
    if (success) {
      setSuccessMessage(`Spending cap updated to $${(pendingCapCents / 100).toFixed(0)}.`);
    }
  }, [pendingCapCents, settings?.spending_cap_cents, clearError, updateSpendingCap]);

  const minCapCents = config?.min_cap_cents ?? DEFAULT_MIN_CAP_CENTS;
  const maxCapCents = config?.max_cap_cents ?? DEFAULT_MAX_CAP_CENTS;
  const settingsCapCents = settings?.spending_cap_cents ?? DEFAULT_CAP_CENTS;
  const currentCapCents = pendingCapCents ?? settingsCapCents;
  const hasCapChanged = pendingCapCents !== null && pendingCapCents !== settingsCapCents;
  const warningThreshold = (config?.warning_threshold ?? DEFAULT_WARNING_THRESHOLD) * 100;

  return {
    subscription,
    config,
    settings,
    isLoading,
    isEnabling,
    isDisabling,
    isUpdatingCap,
    error,
    successMessage,
    pendingCapCents,
    setPendingCapCents,
    handleToggleOverage,
    handleUpdateCap,
    minCapCents,
    maxCapCents,
    currentCapCents,
    hasCapChanged,
    warningThreshold,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-Components
// ─────────────────────────────────────────────────────────────────────────────

interface AlertsSectionProps {
  error: string | null;
  successMessage: string | null;
  showIneligible: boolean;
}

function AlertsSection({ error, successMessage, showIneligible }: AlertsSectionProps) {
  return (
    <>
      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}
      {successMessage && (
        <Alert className="border-success/50 bg-success/5 text-success-foreground">
          <CheckCircle2 className="h-4 w-4" />
          <AlertTitle>Success</AlertTitle>
          <AlertDescription>{successMessage}</AlertDescription>
        </Alert>
      )}
      {showIneligible && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Not Available</AlertTitle>
          <AlertDescription>
            Usage-based pricing requires an active paid subscription with a payment method on file.
          </AlertDescription>
        </Alert>
      )}
    </>
  );
}

interface ToggleSectionProps {
  enabled: boolean;
  marginPercent: number;
  disabled: boolean;
  onToggle: () => void;
}

function ToggleSection({ enabled, marginPercent, disabled, onToggle }: ToggleSectionProps) {
  return (
    <div className="flex items-center justify-between rounded-lg border bg-secondary/40 p-4">
      <div className="space-y-1">
        <p className="text-sm font-medium">Enable usage-based pricing</p>
        <p className="text-xs text-muted-foreground">
          Charges pass-through LLM costs + {Math.round(marginPercent)}% margin
        </p>
      </div>
      <Switch checked={enabled} onCheckedChange={onToggle} disabled={disabled} />
    </div>
  );
}

interface UsageSectionProps {
  currentUsageDollars: number;
  spendingCapDollars: number;
  remainingDollars: number;
  usagePercentage: number;
  capReached: boolean;
  isAtWarning: boolean;
}

function UsageSection({
  currentUsageDollars,
  spendingCapDollars,
  remainingDollars,
  usagePercentage,
  capReached,
  isAtWarning,
}: UsageSectionProps) {
  return (
    <div className="space-y-3 rounded-lg border p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Current Period Usage</p>
        <div className="flex items-center gap-2">
          <DollarSign className="h-4 w-4 text-muted-foreground" />
          <span className="text-lg font-semibold">${currentUsageDollars.toFixed(2)}</span>
          <span className="text-sm text-muted-foreground">/ ${spendingCapDollars.toFixed(0)}</span>
        </div>
      </div>
      <Progress
        value={usagePercentage}
        className={cn(
          "h-2",
          capReached && "bg-destructive/20",
          isAtWarning && !capReached && "bg-warning/20"
        )}
      />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <UsageStatusText
          capReached={capReached}
          isAtWarning={isAtWarning}
          usagePercentage={usagePercentage}
        />
        <span>${remainingDollars.toFixed(2)} remaining</span>
      </div>
    </div>
  );
}

interface UsageStatusTextProps {
  capReached: boolean;
  isAtWarning: boolean;
  usagePercentage: number;
}

function UsageStatusText({ capReached, isAtWarning, usagePercentage }: UsageStatusTextProps) {
  if (capReached) {
    return <span className="text-destructive font-medium">Spending cap reached</span>;
  }
  if (isAtWarning) {
    return (
      <span className="text-warning font-medium">
        Approaching spending cap ({usagePercentage.toFixed(0)}%)
      </span>
    );
  }
  return <span>{usagePercentage.toFixed(0)}% of cap used</span>;
}

interface SpendingCapSectionProps {
  currentCapCents: number;
  minCapCents: number;
  maxCapCents: number;
  hasCapChanged: boolean;
  isUpdatingCap: boolean;
  onCapChange: (cents: number) => void;
  onUpdateCap: () => void;
}

function SpendingCapSection({
  currentCapCents,
  minCapCents,
  maxCapCents,
  hasCapChanged,
  isUpdatingCap,
  onCapChange,
  onUpdateCap,
}: SpendingCapSectionProps) {
  const minDollars = minCapCents / 100;
  const maxDollars = maxCapCents / 100;
  const currentDollars = currentCapCents / 100;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Spending Cap</p>
        <p className="text-sm text-muted-foreground">${currentDollars.toFixed(0)} / month</p>
      </div>
      <Slider
        value={[currentDollars]}
        min={minDollars}
        max={maxDollars}
        step={5}
        onValueChange={([value]) => onCapChange(value * 100)}
        disabled={isUpdatingCap}
        className="py-2"
      />
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>${minDollars}</span>
        <span>${maxDollars}</span>
      </div>
      {hasCapChanged && (
        <Button
          variant="brand"
          size="sm"
          onClick={onUpdateCap}
          disabled={isUpdatingCap}
          className="w-full"
        >
          {isUpdatingCap ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Updating...
            </>
          ) : (
            `Update cap to $${currentDollars.toFixed(0)}`
          )}
        </Button>
      )}
    </div>
  );
}

function InfoBox({ marginPercent }: { marginPercent: number }) {
  return (
    <div className="rounded-lg border border-dashed bg-muted/50 p-3 text-xs text-muted-foreground">
      <p>
        Usage-based pricing is charged at{" "}
        <span className="font-medium text-foreground">
          pass-through cost + {Math.round(marginPercent)}%
        </span>
        . You'll only be charged for usage beyond your subscription's monthly LLM credit allowance,
        up to your spending cap.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────────────────────────────────────

export function OverageSettingsCard() {
  const {
    subscription,
    settings,
    isLoading,
    isEnabling,
    isDisabling,
    isUpdatingCap,
    error,
    successMessage,
    setPendingCapCents,
    handleToggleOverage,
    handleUpdateCap,
    minCapCents,
    maxCapCents,
    currentCapCents,
    hasCapChanged,
    warningThreshold,
  } = useOverageSettings();

  if (subscription?.tier === "free") return null;

  if (isLoading && !settings) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="h-6 w-6 animate-spin text-seer" />
        </CardContent>
      </Card>
    );
  }

  if (!settings) return null;

  const usagePercentage =
    settings.spending_cap_cents > 0
      ? Math.min(100, (settings.current_usage_cents / settings.spending_cap_cents) * 100)
      : 0;
  const isAtWarning = usagePercentage >= warningThreshold;
  const toggleDisabled =
    (!settings.eligible && !settings.enabled) || isEnabling || isDisabling || isLoading;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-seer/10">
            <TrendingUp className="h-5 w-5 text-seer" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">Usage-Based Pricing</CardTitle>
              {settings.enabled && (
                <Badge className="bg-success/10 text-success border-success/20">Active</Badge>
              )}
            </div>
            <CardDescription>
              Pay for additional LLM credits beyond your subscription allowance
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <AlertsSection
          error={error}
          successMessage={successMessage}
          showIneligible={!settings.eligible && !settings.enabled}
        />
        <ToggleSection
          enabled={settings.enabled}
          marginPercent={settings.margin_percent}
          disabled={toggleDisabled}
          onToggle={handleToggleOverage}
        />
        {settings.enabled && (
          <UsageSection
            currentUsageDollars={settings.current_usage_dollars}
            spendingCapDollars={settings.spending_cap_dollars}
            remainingDollars={settings.remaining_dollars}
            usagePercentage={usagePercentage}
            capReached={settings.cap_reached}
            isAtWarning={isAtWarning}
          />
        )}
        <SpendingCapSection
          currentCapCents={currentCapCents}
          minCapCents={minCapCents}
          maxCapCents={maxCapCents}
          hasCapChanged={hasCapChanged}
          isUpdatingCap={isUpdatingCap}
          onCapChange={setPendingCapCents}
          onUpdateCap={handleUpdateCap}
        />
        <InfoBox marginPercent={settings.margin_percent} />
      </CardContent>
    </Card>
  );
}
