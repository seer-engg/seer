import { Bar, BarChart, XAxis, YAxis, CartesianGrid, Cell } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartContainer, ChartTooltip, type ChartConfig } from "@/components/ui/chart";
import { Skeleton } from "@/components/ui/skeleton";
import type { ModelBreakdownItem } from "@/lib/usage-analytics-api";

const MODEL_COLORS = [
  "hsl(239 84% 67%)",   // seer purple
  "hsl(160 84% 39%)",   // success emerald
  "hsl(38 92% 50%)",    // warning amber
  "hsl(199 89% 48%)",   // sky blue
  "hsl(347 77% 50%)",   // bug red
  "hsl(271 81% 56%)",   // violet
  "hsl(142 71% 45%)",   // green
  "hsl(25 95% 53%)",    // orange
];

const chartConfig = {
  total_cost: {
    label: "Cost",
  },
} satisfies ChartConfig;

function truncateModel(name: string, maxLen = 24): string {
  if (name.length <= maxLen) return name;
  return name.slice(0, maxLen - 1) + "\u2026";
}

interface ModelBreakdownChartProps {
  data?: ModelBreakdownItem[];
  isLoading: boolean;
}

export function ModelBreakdownChart({ data, isLoading }: ModelBreakdownChartProps) {
  const chartHeight = Math.max(200, (data?.length ?? 0) * 40 + 40);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Cost by Model</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <Skeleton className="h-[200px] w-full rounded-lg" />
        ) : !data?.length ? (
          <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
            No model usage data for this period
          </div>
        ) : (
          <ChartContainer config={chartConfig} className="w-full" style={{ height: chartHeight }}>
            <BarChart data={data} layout="vertical" margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis
                type="number"
                tickLine={false}
                axisLine={false}
                tickFormatter={(value: number) => `$${value.toFixed(2)}`}
                tick={{ fontSize: 11 }}
              />
              <YAxis
                type="category"
                dataKey="model"
                tickLine={false}
                axisLine={false}
                width={140}
                tickFormatter={truncateModel}
                tick={{ fontSize: 11 }}
              />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const item = payload[0].payload as ModelBreakdownItem;
                  return (
                    <div className="rounded-lg border border-border/50 bg-background px-3 py-2 text-xs shadow-xl space-y-1">
                      <p className="font-medium">{item.model}</p>
                      <p className="text-muted-foreground">Provider: {item.provider}</p>
                      <p>Cost: <span className="font-mono font-medium">${item.total_cost.toFixed(4)}</span></p>
                      <p>Tokens: <span className="font-mono">{item.total_tokens.toLocaleString()}</span></p>
                      <p>Calls: <span className="font-mono">{item.call_count.toLocaleString()}</span></p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="total_cost" radius={[0, 4, 4, 0]}>
                {data.map((_, index) => (
                  <Cell key={index} fill={MODEL_COLORS[index % MODEL_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ChartContainer>
        )}
      </CardContent>
    </Card>
  );
}
