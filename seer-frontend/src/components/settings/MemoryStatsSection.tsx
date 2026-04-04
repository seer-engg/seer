import { Badge } from "@/components/ui/badge";
import { type MemoryStats } from "@/lib/memory-api";

interface MemoryStatsSectionProps {
  stats: MemoryStats;
}

function StatItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

export function MemoryStatsSection({ stats }: MemoryStatsSectionProps) {
  return (
    <div className="space-y-3 p-3 rounded-lg bg-muted/30 border">
      <StatItem label="Total memories" value={stats.total_memories} />
      <div className="flex items-center justify-between text-sm pt-1 border-t">
        <span className="text-muted-foreground">Feature status</span>
        <Badge
          variant="secondary"
          className={
            stats.memory_enabled
              ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
              : "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20"
          }
        >
          {stats.memory_enabled ? "Enabled" : "Disabled"}
        </Badge>
      </div>
    </div>
  );
}
