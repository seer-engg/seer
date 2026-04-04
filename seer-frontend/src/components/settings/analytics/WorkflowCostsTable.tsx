import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useWorkflowAnalytics } from "@/hooks/useUsageAnalytics";
import type { WorkflowCostItem } from "@/lib/usage-analytics-api";
import type { DateRange } from "./DateRangeFilter";

type SortKey = "workflow_name" | "total_cost" | "call_count";
type SortDir = "asc" | "desc";

interface WorkflowCostsTableProps {
  dateRange: DateRange;
}

export function WorkflowCostsTable({ dateRange }: WorkflowCostsTableProps) {
  const { data, isLoading } = useWorkflowAnalytics(dateRange);
  const [sortKey, setSortKey] = useState<SortKey>("total_cost");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const sorted = useMemo(() => {
    if (!data?.workflows) return [];
    return [...data.workflows].sort((a, b) => {
      let cmp: number;
      if (sortKey === "workflow_name") {
        cmp = (a.workflow_name ?? a.workflow_id).localeCompare(b.workflow_name ?? b.workflow_id);
      } else {
        cmp = (a[sortKey] as number) - (b[sortKey] as number);
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
  }, [data?.workflows, sortKey, sortDir]);

  const columns: { key: SortKey; label: string; className?: string }[] = [
    { key: "workflow_name", label: "Workflow" },
    { key: "total_cost", label: "Total Cost", className: "text-right" },
    { key: "call_count", label: "Calls", className: "text-right" },
  ];

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Cost by Workflow</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-hidden rounded-lg border border-border/60">
          <Table>
            <TableHeader>
              <TableRow>
                {columns.map((col) => (
                  <TableHead
                    key={col.key}
                    className={cn("cursor-pointer select-none hover:text-foreground", col.className)}
                    onClick={() => handleSort(col.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.label}
                      <ArrowUpDown className={cn(
                        "h-3 w-3",
                        sortKey === col.key ? "text-foreground" : "text-muted-foreground/50"
                      )} />
                    </span>
                  </TableHead>
                ))}
                <TableHead className="text-right">Tokens</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading
                ? Array.from({ length: 4 }).map((_, idx) => (
                    <TableRow key={`wf-skeleton-${idx}`}>
                      <TableCell><Skeleton className="h-4 w-32" /></TableCell>
                      <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-16" /></TableCell>
                      <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-12" /></TableCell>
                      <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-20" /></TableCell>
                    </TableRow>
                  ))
                : null}
              {!isLoading && sorted.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={4} className="py-10 text-center text-muted-foreground">
                    No workflow costs for this period
                  </TableCell>
                </TableRow>
              ) : null}
              {!isLoading
                ? sorted.map((wf: WorkflowCostItem) => (
                    <TableRow key={wf.workflow_id}>
                      <TableCell className="font-medium">
                        {wf.workflow_name ?? wf.workflow_id}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        ${wf.total_cost.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {wf.call_count.toLocaleString()}
                      </TableCell>
                      <TableCell className="text-right font-mono text-sm">
                        {wf.total_tokens.toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))
                : null}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
