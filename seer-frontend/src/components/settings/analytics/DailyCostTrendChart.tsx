import { Area, AreaChart, XAxis, YAxis, CartesianGrid } from "recharts";
import { format, parseISO } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import type { DailyTrendItem } from "@/lib/usage-analytics-api";

const chartConfig = {
  total_cost: {
    label: "Cost",
    color: "hsl(239 84% 67%)",
  },
} satisfies ChartConfig;

interface DailyCostTrendChartProps {
  data?: DailyTrendItem[];
  isLoading: boolean;
}

export function DailyCostTrendChart({ data, isLoading }: DailyCostTrendChartProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Daily Cost Trend</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-[250px] w-full rounded-lg" />
        ) : !data?.length ? (
          <div className="flex h-[250px] items-center justify-center text-sm text-muted-foreground">
            No usage data for this period
          </div>
        ) : (
          <ChartContainer config={chartConfig} className="h-[250px] w-full">
            <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickFormatter={(value: string) => {
                  try {
                    return format(parseISO(value), "MMM d");
                  } catch {
                    return value;
                  }
                }}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                tickLine={false}
                axisLine={false}
                tickFormatter={(value: number) => `$${value.toFixed(2)}`}
                tick={{ fontSize: 11 }}
                width={50}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    formatter={(value) => `$${Number(value).toFixed(4)}`}
                  />
                }
              />
              <Area
                type="monotone"
                dataKey="total_cost"
                stroke="var(--color-total_cost)"
                fill="var(--color-total_cost)"
                fillOpacity={0.15}
                strokeWidth={2}
              />
            </AreaChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
