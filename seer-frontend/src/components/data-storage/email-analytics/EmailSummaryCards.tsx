import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Send, Eye, MousePointerClick, Percent, TrendingUp } from "lucide-react";
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

interface EmailSummaryCardsProps {
  totalSent?: number;
  totalOpened?: number;
  totalClicked?: number;
  openRate?: number;
  clickRate?: number;
  isLoading: boolean;
}

export function EmailSummaryCards({ totalSent, totalOpened, totalClicked, openRate, clickRate, isLoading }: EmailSummaryCardsProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-5 gap-4">
      <SummaryCard
        label="Sent"
        value={(totalSent ?? 0).toLocaleString()}
        icon={<Send className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-blue-500/70 to-blue-400/70"
        isLoading={isLoading}
      />
      <SummaryCard
        label="Opened"
        value={(totalOpened ?? 0).toLocaleString()}
        icon={<Eye className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-emerald-500/70 to-teal-400/70"
        isLoading={isLoading}
      />
      <SummaryCard
        label="Clicked"
        value={(totalClicked ?? 0).toLocaleString()}
        icon={<MousePointerClick className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-purple-500/70 to-violet-400/70"
        isLoading={isLoading}
      />
      <SummaryCard
        label="Open Rate"
        value={`${(openRate ?? 0).toFixed(1)}%`}
        icon={<Percent className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-amber-500/70 to-orange-400/70"
        isLoading={isLoading}
      />
      <SummaryCard
        label="Click Rate"
        value={`${(clickRate ?? 0).toFixed(1)}%`}
        icon={<TrendingUp className="h-5 w-5" />}
        accentClass="bg-gradient-to-br from-rose-500/70 to-pink-400/70"
        isLoading={isLoading}
      />
    </div>
  );
}
