import { useState } from "react";
import { format, parseISO } from "date-fns";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { cn } from "@/lib/utils";
import { useUsageRecords } from "@/hooks/useUsageAnalytics";
import type { UsageRecord } from "@/lib/usage-analytics-api";
import type { DateRange } from "./DateRangeFilter";

const PAGE_SIZE = 50;

const OPERATION_LABELS: Record<string, string> = {
  workflow_execution: "Workflow",
  chat_message: "Chat",
  browser_execution: "Browser",
};

const OPERATION_STYLES: Record<string, string> = {
  workflow_execution: "bg-seer/15 text-seer border-seer/30",
  chat_message: "bg-success/15 text-success border-success/40",
  browser_execution: "bg-warning/15 text-warning border-warning/30",
};

function formatDateTime(iso: string): string {
  try {
    return format(parseISO(iso), "MMM d, yyyy HH:mm");
  } catch {
    return iso;
  }
}

function RecordSkeletonRows() {
  return (
    <>
      {Array.from({ length: 5 }).map((_, idx) => (
        <TableRow key={`rec-skeleton-${idx}`}>
          <TableCell><Skeleton className="h-4 w-32" /></TableCell>
          <TableCell><Skeleton className="h-4 w-16" /></TableCell>
          <TableCell><Skeleton className="h-4 w-28" /></TableCell>
          <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-16" /></TableCell>
          <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-16" /></TableCell>
          <TableCell className="text-right"><Skeleton className="ml-auto h-4 w-14" /></TableCell>
          <TableCell><Skeleton className="h-4 w-16" /></TableCell>
        </TableRow>
      ))}
    </>
  );
}

function RecordRow({ rec, isFetching }: { rec: UsageRecord; isFetching: boolean }) {
  const opLabel = OPERATION_LABELS[rec.operation] ?? rec.operation.replace(/_/g, " ");
  return (
    <TableRow className={cn(isFetching && "opacity-60 transition-opacity")}>
      <TableCell className="text-sm whitespace-nowrap">{formatDateTime(rec.created_at)}</TableCell>
      <TableCell className="text-sm">{rec.provider}</TableCell>
      <TableCell className="text-sm font-mono">{rec.model}</TableCell>
      <TableCell className="text-right font-mono text-sm">{rec.input_tokens.toLocaleString()}</TableCell>
      <TableCell className="text-right font-mono text-sm">{rec.output_tokens.toLocaleString()}</TableCell>
      <TableCell className="text-right font-mono text-sm">${rec.cost.toFixed(4)}</TableCell>
      <TableCell>
        <Badge variant="outline" className={cn("capitalize text-[10px]", OPERATION_STYLES[rec.operation] ?? "bg-muted/70")}>
          {opLabel}
        </Badge>
      </TableCell>
    </TableRow>
  );
}

function RecordsPagination({ page, hasMore, onPageChange }: { page: number; hasMore: boolean; onPageChange: (p: number) => void }) {
  return (
    <Pagination className="pt-4">
      <PaginationContent>
        <PaginationItem>
          <PaginationPrevious href="#" onClick={(e) => { e.preventDefault(); if (page > 1) onPageChange(page - 1); }} className={cn(page === 1 ? "pointer-events-none opacity-50" : "")} />
        </PaginationItem>
        {page > 1 && (
          <PaginationItem>
            <PaginationLink href="#" onClick={(e) => { e.preventDefault(); onPageChange(page - 1); }}>{page - 1}</PaginationLink>
          </PaginationItem>
        )}
        <PaginationItem>
          <PaginationLink href="#" isActive>{page}</PaginationLink>
        </PaginationItem>
        {hasMore && (
          <PaginationItem>
            <PaginationLink href="#" onClick={(e) => { e.preventDefault(); onPageChange(page + 1); }}>{page + 1}</PaginationLink>
          </PaginationItem>
        )}
        <PaginationItem>
          <PaginationNext href="#" onClick={(e) => { e.preventDefault(); if (hasMore) onPageChange(page + 1); }} className={cn(!hasMore ? "pointer-events-none opacity-50" : "")} />
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  );
}

interface UsageRecordsTableProps {
  dateRange: DateRange;
}

export function UsageRecordsTable({ dateRange }: UsageRecordsTableProps) {
  const [page, setPage] = useState(1);
  const offset = (page - 1) * PAGE_SIZE;
  const { data, isLoading, isFetching } = useUsageRecords({ ...dateRange, limit: PAGE_SIZE, offset });

  const records = data?.records ?? [];
  const total = data?.total ?? 0;
  const hasMore = offset + PAGE_SIZE < total;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) setPage(newPage);
  };

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-base">Usage Records</CardTitle>
        {total > 0 && <span className="text-xs text-muted-foreground">{total.toLocaleString()} total records</span>}
      </CardHeader>
      <CardContent>
        <div className="overflow-hidden rounded-lg border border-border/60">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Date/Time</TableHead>
                  <TableHead>Provider</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead className="text-right">Input Tokens</TableHead>
                  <TableHead className="text-right">Output Tokens</TableHead>
                  <TableHead className="text-right">Cost</TableHead>
                  <TableHead>Operation</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading && records.length === 0 ? <RecordSkeletonRows /> : null}
                {!isLoading && records.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="py-10 text-center text-muted-foreground">No usage records for this period</TableCell>
                  </TableRow>
                ) : null}
                {records.map((rec) => <RecordRow key={rec.id} rec={rec} isFetching={isFetching} />)}
              </TableBody>
            </Table>
          </div>
        </div>
        {totalPages > 1 && <RecordsPagination page={page} hasMore={hasMore} onPageChange={handlePageChange} />}
      </CardContent>
    </Card>
  );
}
