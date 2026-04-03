import { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { useUsageAnalytics } from "@/hooks/useUsageAnalytics";
import { DateRangeFilter, type DateRange } from "./DateRangeFilter";
import { AnalyticsSummaryCards } from "./AnalyticsSummaryCards";
import { DailyCostTrendChart } from "./DailyCostTrendChart";
import { ModelBreakdownChart } from "./ModelBreakdownChart";
import { OperationBreakdownChart } from "./OperationBreakdownChart";
import { WorkflowCostsTable } from "./WorkflowCostsTable";
import { UsageRecordsTable } from "./UsageRecordsTable";

export function UsageAnalyticsTab() {
  const [activePreset, setActivePreset] = useState("billing");
  const [dateRange, setDateRange] = useState<DateRange>({});

  const { data, isLoading, error } = useUsageAnalytics(dateRange);

  const handleDateRangeChange = (key: string, range: DateRange) => {
    setActivePreset(key);
    setDateRange(range);
  };

  const totalTokens = data?.total_tokens;

  const periodLabel = data
    ? `${new Date(data.period_start).toLocaleDateString(undefined, { month: "short", day: "numeric" })} – ${new Date(data.period_end).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}`
    : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <DateRangeFilter activeKey={activePreset} onChange={handleDateRangeChange} />
        {periodLabel && (
          <span className="text-xs text-muted-foreground">{periodLabel}</span>
        )}
      </div>

      {error ? (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Failed to load analytics</AlertTitle>
          <AlertDescription>
            {error instanceof Error ? error.message : "An unexpected error occurred."}
          </AlertDescription>
        </Alert>
      ) : null}

      <AnalyticsSummaryCards
        totalCost={data?.total_cost}
        totalTokens={totalTokens}
        totalCalls={data?.total_calls}
        isLoading={isLoading}
      />

      <DailyCostTrendChart data={data?.daily_trend} isLoading={isLoading} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ModelBreakdownChart data={data?.by_model} isLoading={isLoading} />
        <OperationBreakdownChart data={data?.by_operation} isLoading={isLoading} />
      </div>

      <WorkflowCostsTable dateRange={dateRange} />

      <UsageRecordsTable dateRange={dateRange} />
    </div>
  );
}
