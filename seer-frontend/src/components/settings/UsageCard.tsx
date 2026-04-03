import { useEffect, useMemo, type ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Activity,
  Calendar,
  Clock,
  RefreshCw,
  Rocket,
  Sparkles,
  Info,
  Users,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn, formatRelativeCountdown } from "@/lib/utils";
import { useUsageStore } from "@/stores/usageStore";
import { useOrganizationStore } from "@/stores/organizationStore";
import { useSubscriptionStore } from "@/stores/subscriptionStore";
import type { UsageMetric, UsageSummary } from "@/lib/usage-api";

const USD_PRECISION = 2;

interface MetricConfig {
  key: string;
  label: string;
  metric?: UsageMetric;
  icon: ReactNode;
  accent: string;
  useRelativeCountdown?: boolean;
}

interface UsageMeterProps {
  label: string;
  metric?: UsageMetric;
  icon?: ReactNode;
  accentClass: string;
  useRelativeCountdown?: boolean;
}

function formatValue(value: number, unit?: UsageMetric["unit"]) {
  return unit === "usd" ? value.toFixed(USD_PRECISION) : value.toFixed(0);
}

function getLimitValue(metric: UsageMetric) {
  if (metric.limit === null || metric.limit === undefined) return null;
  return formatValue(metric.limit, metric.unit);
}

function getLimitLabel(metric: UsageMetric, usedLabel: string, limitValue: string | null) {
  if (metric.disabled) return "Disabled";
  if (metric.is_unlimited || limitValue === null) return "Unlimited";
  return `${usedLabel}/${limitValue}`;
}

function getRemainingLabel(metric: UsageMetric) {
  if (metric.disabled || metric.is_unlimited || metric.limit === null) return null;
  return `${Math.max(metric.remaining ?? 0, 0).toFixed(metric.unit === "usd" ? USD_PRECISION : 0)} left`;
}

function getProgress(metric: UsageMetric) {
  const hasCap = metric.limit !== null && metric.limit !== undefined && !metric.is_unlimited;
  if (!hasCap || !metric.limit || metric.limit <= 0) return 0;
  return Math.min(100, (metric.used / metric.limit) * 100);
}

function getResetLabel(metric: UsageMetric, useRelativeCountdown = false) {
  if (metric.reset_at) {
    if (useRelativeCountdown) {
      const countdown = formatRelativeCountdown(metric.reset_at);
      return countdown ? `Resets ${countdown}` : null;
    }
    return `Resets ${new Date(metric.reset_at).toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;
  }
  if (metric.disabled) return "Not available";
  if (metric.is_unlimited) return "No cap";
  return null;
}

function UsageMeterSkeleton() {
  return (
    <div className="space-y-2 rounded-lg border bg-secondary/40 p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Skeleton className="h-4 w-4 rounded-full" />
          <Skeleton className="h-3 w-24" />
        </div>
        <Skeleton className="h-3 w-16" />
      </div>
      <Skeleton className="h-2.5 w-full rounded-full" />
    </div>
  );
}

function UsageMeter({ label, metric, icon, accentClass, useRelativeCountdown = false }: UsageMeterProps) {
  if (!metric) return <UsageMeterSkeleton />;

  const usedLabel = formatValue(metric.used, metric.unit);
  const limitValue = getLimitValue(metric);
  const limitLabel = getLimitLabel(metric, usedLabel, limitValue);
  const remainingLabel = getRemainingLabel(metric);
  const progress = getProgress(metric);
  const resetLabel = getResetLabel(metric, useRelativeCountdown);

  return (
    <div className="space-y-2 rounded-lg border bg-secondary/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <div className={cn("h-9 w-9 rounded-lg grid place-items-center text-white", accentClass)}>
            {icon}
          </div>
          <div className="flex flex-col leading-tight">
            <span className="text-xs uppercase tracking-[0.08em] text-muted-foreground">{label}</span>
            <span className="text-sm font-semibold text-foreground">{limitLabel}</span>
          </div>
        </div>
        {remainingLabel && (
          <span className="text-xs font-medium text-muted-foreground">{remainingLabel}</span>
        )}
      </div>
      <Progress value={progress} className="h-2.5" />
      {resetLabel && <p className="text-[11px] text-muted-foreground">{resetLabel}</p>}
    </div>
  );
}

function UsageCardHeader({
  isLoading,
  onRefresh,
  isTeamOrg,
  memberCount,
}: {
  isLoading: boolean;
  onRefresh: () => void;
  isTeamOrg: boolean;
  memberCount?: number;
}) {
  const subscription = useSubscriptionStore((state) => state.subscription);
  const tier = subscription?.tier ?? 'free';

  return (
    <CardHeader>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-seer" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">
                {isTeamOrg ? 'Team Usage' : 'Usage & Limits'}
              </CardTitle>
              {isTeamOrg && (
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="inline-flex items-center gap-1 cursor-help">
                        <Info className="h-3.5 w-3.5 text-muted-foreground" />
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="right" className="max-w-xs">
                      <p>Usage is shared across all {memberCount ?? 0} team members. Everyone contributes to and draws from the same quota pool.</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              )}
            </div>
            <CardDescription>
              {isTeamOrg
                ? `Shared quota across ${memberCount ?? 0} members`
                : 'Live quota snapshot'
              }
            </CardDescription>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {subscription ? (
            <Badge variant="secondary" className="h-6 px-2 bg-seer/15 text-seer border-seer/30">
              {tier.replace("_", " ")}
            </Badge>
          ) : (
            <Skeleton className="h-6 w-16 rounded-full" />
          )}
          <Button
            variant="outline"
            size="icon"
            className="h-9 w-9"
            onClick={onRefresh}
            title="Refresh usage"
            disabled={isLoading}
          >
            <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
          </Button>
        </div>
      </div>
    </CardHeader>
  );
}

function UsageMeters({ metrics }: { metrics: MetricConfig[] }) {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {metrics.length === 0
        ? Array.from({ length: 2 }).map((_, idx) => <UsageMeter key={idx} label="" accentClass="bg-secondary" />)
        : metrics.map(({ key, label, metric, icon, accent, useRelativeCountdown }) => (
            <UsageMeter key={key} label={label} metric={metric} icon={icon} accentClass={accent} useRelativeCountdown={useRelativeCountdown} />
          ))}
    </div>
  );
}

function AICreditsGroup({ metrics }: { metrics: MetricConfig[] }) {
  if (metrics.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-4 space-y-3">
        <div className="flex items-center gap-2 text-sm font-medium text-foreground">
          <Sparkles className="h-4 w-4 text-sky-500" />
          AI Credits
        </div>
        <div className="space-y-2">
          {Array.from({ length: 3 }).map((_, idx) => <UsageMeterSkeleton key={idx} />)}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
        <Sparkles className="h-4 w-4 text-sky-500" />
        AI Credits
      </div>
      <div className="space-y-2">
        {metrics.map(({ key, label, metric, icon, accent, useRelativeCountdown }) => (
          <UsageMeter key={key} label={label} metric={metric} icon={icon} accentClass={accent} useRelativeCountdown={useRelativeCountdown} />
        ))}
      </div>
    </div>
  );
}

function UsageMeta({ usage }: { usage: UsageSummary | null }) {
  const subscription = useSubscriptionStore((state) => state.subscription);

  if (!usage) return null;

  const status = subscription?.status ?? 'active';

  return (
    <div className="flex flex-col gap-1 text-[12px] text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
      <span>
        {status === "trialing"
          ? "Trial active"
          : usage.is_self_hosted
            ? "Self-hosted mode"
            : `Billing: ${status}`}
      </span>
      <span>Min poll: {usage.limits.poll_min_interval_seconds}s</span>
    </div>
  );
}

export function UsageCard() {
  const usage = useUsageStore((state) => state.usage);
  const isLoading = useUsageStore((state) => state.isLoading);
  const error = useUsageStore((state) => state.error);
  const fetchUsage = useUsageStore((state) => state.fetchUsage);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const isTeamOrg = currentOrganization?.type === 'team';
  const memberCount = currentOrganization?.memberCount;

  useEffect(() => {
    if (!usage && !isLoading) {
      void fetchUsage();
    }
  }, [usage, isLoading, fetchUsage]);

  // Top row metrics: Workflow runs and Workflows
  const topMetrics = useMemo(() => {
    if (!usage) return [];
    return [
      {
        key: "runs",
        label: "Workflow runs",
        metric: usage.usage.workflow_runs,
        icon: <Activity className="h-4 w-4" />,
        accent: "bg-gradient-to-br from-emerald-500/70 to-teal-400/70",
      },
      {
        key: "workflows",
        label: "Workflows",
        metric: usage.usage.workflows,
        icon: <Rocket className="h-4 w-4" />,
        accent: "bg-gradient-to-br from-amber-500/70 to-orange-400/70",
      },
    ];
  }, [usage]);

  // AI Credits group: 5-hour, weekly, and monthly
  const aiCreditsMetrics = useMemo(() => {
    if (!usage) return [];
    return [
      {
        key: "llm_5h",
        label: "5-Hour",
        metric: usage.usage.llm_credits_5h,
        icon: <Clock className="h-4 w-4" />,
        accent: "bg-gradient-to-br from-violet-500/70 to-purple-400/70",
        useRelativeCountdown: true,
      },
      {
        key: "llm_weekly",
        label: "Weekly",
        metric: usage.usage.llm_credits_weekly,
        icon: <Calendar className="h-4 w-4" />,
        accent: "bg-gradient-to-br from-sky-500/70 to-cyan-400/70",
        useRelativeCountdown: true,
      },
      {
        key: "llm_monthly",
        label: "Monthly",
        metric: usage.usage.llm_credits,
        icon: <Sparkles className="h-4 w-4" />,
        accent: "bg-gradient-to-br from-sky-500/70 to-indigo-400/60",
        useRelativeCountdown: false,
      },
    ];
  }, [usage]);

  return (
    <Card>
      <UsageCardHeader
        isLoading={isLoading}
        onRefresh={() => fetchUsage({ force: true })}
        isTeamOrg={isTeamOrg}
        memberCount={memberCount}
      />
      <CardContent className="space-y-4">
        {error && (
          <div className="text-xs text-amber-600 bg-amber-500/10 border border-amber-500/30 rounded-md p-2">
            {error}
          </div>
        )}
        <UsageMeters metrics={topMetrics} />
        <AICreditsGroup metrics={aiCreditsMetrics} />
        <UsageMeta usage={usage} />

        {/* Team usage breakdown link */}
        {isTeamOrg && (
          <div className="pt-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start text-muted-foreground hover:text-foreground"
              onClick={() => {
                // Navigate to analytics tab which has detailed breakdown
                window.location.href = '/settings?tab=analytics';
              }}
            >
              <Users className="h-4 w-4 mr-2" />
              View usage breakdown by member
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
