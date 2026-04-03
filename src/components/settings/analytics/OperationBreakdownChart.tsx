import { Pie, PieChart } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import type { OperationBreakdownItem } from "@/lib/usage-analytics-api";

const OPERATION_LABELS: Record<string, string> = {
  workflow_execution: "Workflows",
  chat_message: "Chat",
  browser_execution: "Browser",
};

const OPERATION_COLORS: Record<string, string> = {
  workflow_execution: "hsl(239 84% 67%)",
  chat_message: "hsl(160 84% 39%)",
  browser_execution: "hsl(38 92% 50%)",
};

function getOperationLabel(op: string): string {
  return OPERATION_LABELS[op] ?? op.replace(/_/g, " ");
}

function buildChartConfig(data: OperationBreakdownItem[]): ChartConfig {
  const config: ChartConfig = {};
  for (const item of data) {
    config[item.operation] = {
      label: getOperationLabel(item.operation),
      color: OPERATION_COLORS[item.operation] ?? "hsl(var(--muted-foreground))",
    };
  }
  return config;
}

interface OperationBreakdownChartProps {
  data?: OperationBreakdownItem[];
  isLoading: boolean;
}

export function OperationBreakdownChart({ data, isLoading }: OperationBreakdownChartProps) {
  const chartData = data?.map((item) => ({
    ...item,
    fill: OPERATION_COLORS[item.operation] ?? "hsl(var(--muted-foreground))",
  }));

  const chartConfig = data ? buildChartConfig(data) : {};

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Cost by Operation</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-[200px] w-full rounded-lg" />
        ) : !chartData?.length ? (
          <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
            No operation data for this period
          </div>
        ) : (
          <ChartContainer config={chartConfig} className="mx-auto h-[250px] w-full">
            <PieChart>
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    nameKey="operation"
                    formatter={(value) => `$${Number(value).toFixed(4)}`}
                  />
                }
              />
              <Pie
                data={chartData}
                dataKey="total_cost"
                nameKey="operation"
                innerRadius={60}
                outerRadius={90}
                strokeWidth={2}
                stroke="hsl(var(--background))"
              />
              <ChartLegend
                content={<ChartLegendContent nameKey="operation" />}
              />
            </PieChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
