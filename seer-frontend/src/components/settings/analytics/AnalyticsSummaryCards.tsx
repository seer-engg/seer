import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { DollarSign, Hash, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface SummaryCardProps {
  label: string;
  value: string;
  icon: ReactNode;
  accentClass: string;
  isLoading?: boolean;
}

function SummaryCard({ label, value, icon, accentClass, isLoading }: SummaryCardProps) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4">
        <div className={cn("h-10 w-10 rounded-lg grid place-items-center text-white shrink-0", accentClass)}>
          {icon}
        </div>
        <div className="flex flex-col leading-tight min-w-0">
          <span className="text-xs uppercase tracking-[0.08em] text-muted-foreground">{label}</span>
          {isLoading ? (
            <Skeleton className="h-5 w-20 mt-0.5" />
          ) : (
            <span className="text-lg font-semibold text-foreground truncate">{value}</span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

interface AnalyticsSummaryCardsProps {
  totalCost?: number;
  totalTokens?: number;
  totalCalls?: number;
  isLoading: boolean;
}

export function AnalyticsSummaryCards({ totalCost, totalTokens, totalCalls, isLoading }: AnalyticsSummaryCardsProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      <SummaryCard
        label="Total Cost"
        value={`$${(totalCost ?? 0).toFixed(2)}`}
        icon={<DollarSign className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-emerald-500/70 to-teal-400/70"
        isLoading={isLoading}
      />
      <SummaryCard
        label="Total Tokens"
        value={(totalTokens ?? 0).toLocaleString()}
        icon={<Hash className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-[#3b1d6b]/80 to-[#5b21b6]/60"
        isLoading={isLoading}
      />
      <SummaryCard
        label="API Calls"
        value={(totalCalls ?? 0).toLocaleString()}
        icon={<Activity className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-sky-500/70 to-indigo-400/60"
        isLoading={isLoading}
      />
    </div>
  );
}
