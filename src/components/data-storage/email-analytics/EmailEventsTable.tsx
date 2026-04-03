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
import { useEmailEvents } from "@/hooks/useEmailAnalytics";
import type { EmailEventItem } from "@/lib/email-analytics-api";

const PAGE_SIZE = 50;

const EVENT_TYPE_STYLES: Record<string, string> = {
  sent: "bg-blue-500/15 text-blue-600 border-blue-500/30",
  opened: "bg-emerald-500/15 text-emerald-600 border-emerald-500/30",
  clicked: "bg-purple-500/15 text-purple-600 border-purple-500/30",
};

const EMAIL_TYPE_STYLES: Record<string, string> = {
  invitation: "bg-sky-500/15 text-sky-600 border-sky-500/30",
  approval: "bg-amber-500/15 text-amber-600 border-amber-500/30",
  member_joined: "bg-teal-500/15 text-teal-600 border-teal-500/30",
  workflow: "bg-violet-500/15 text-violet-600 border-violet-500/30",
  hitl: "bg-rose-500/15 text-rose-600 border-rose-500/30",
};

function formatDateTime(iso: string): string {
  try {
    return format(parseISO(iso), "MMM d, yyyy HH:mm");
  } catch {
    return iso;
  }
}

function SkeletonRows() {
  return (
    <>
      {Array.from({ length: 5 }).map((_, idx) => (
        <TableRow key={`skel-${idx}`}>
          <TableCell><Skeleton className="h-4 w-32" /></TableCell>
          <TableCell><Skeleton className="h-4 w-16" /></TableCell>
          <TableCell><Skeleton className="h-4 w-16" /></TableCell>
          <TableCell><Skeleton className="h-4 w-40" /></TableCell>
          <TableCell><Skeleton className="h-4 w-32" /></TableCell>
          <TableCell><Skeleton className="h-4 w-20" /></TableCell>
        </TableRow>
      ))}
    </>
  );
}

function EventRow({ event, isFetching }: { event: EmailEventItem; isFetching: boolean }) {
  return (
    <TableRow className={cn(isFetching && "opacity-60 transition-opacity")}>
      <TableCell className="text-sm whitespace-nowrap">{formatDateTime(event.created_at)}</TableCell>
      <TableCell>
        <Badge variant="outline" className={cn("capitalize text-[10px]", EMAIL_TYPE_STYLES[event.email_type] ?? "bg-muted/70")}>
          {event.email_type.replace(/_/g, " ")}
        </Badge>
      </TableCell>
      <TableCell>
        <Badge variant="outline" className={cn("capitalize text-[10px]", EVENT_TYPE_STYLES[event.event_type] ?? "bg-muted/70")}>
          {event.event_type}
        </Badge>
      </TableCell>
      <TableCell className="text-sm font-mono truncate max-w-[200px]" title={event.recipient}>
        {event.recipient}
      </TableCell>
      <TableCell className="text-sm truncate max-w-[200px]" title={event.subject ?? ""}>
        {event.subject ?? "—"}
      </TableCell>
      <TableCell className="text-sm text-muted-foreground font-mono truncate max-w-[120px]" title={event.workflow_run_id ?? ""}>
        {event.workflow_run_id ?? "—"}
      </TableCell>
    </TableRow>
  );
}

interface EmailEventsTableProps {
  emailType?: string;
  eventType?: string;
}

export function EmailEventsTable({ emailType, eventType }: EmailEventsTableProps) {
  const [page, setPage] = useState(1);
  const offset = (page - 1) * PAGE_SIZE;
  const { data, isLoading, isFetching } = useEmailEvents({
    limit: PAGE_SIZE,
    offset,
    email_type: emailType,
    event_type: eventType,
  });

  const events = data?.events ?? [];
  const total = data?.total ?? 0;
  const hasMore = offset + PAGE_SIZE < total;
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) setPage(newPage);
  };

  return (
    <Card>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-base">Email Events</CardTitle>
        {total > 0 && <span className="text-xs text-muted-foreground">{total.toLocaleString()} total events</span>}
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-[160px]">Time</TableHead>
                <TableHead className="w-[100px]">Type</TableHead>
                <TableHead className="w-[80px]">Event</TableHead>
                <TableHead>Recipient</TableHead>
                <TableHead>Subject</TableHead>
                <TableHead className="w-[120px]">Run ID</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                <SkeletonRows />
              ) : events.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-12">
                    No email events yet. Events will appear here once emails are sent.
                  </TableCell>
                </TableRow>
              ) : (
                events.map((event) => <EventRow key={event.id} event={event} isFetching={isFetching} />)
              )}
            </TableBody>
          </Table>
        </div>
        {totalPages > 1 && (
          <div className="px-4 pb-4">
            <Pagination className="pt-4">
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious href="#" onClick={(e) => { e.preventDefault(); if (page > 1) handlePageChange(page - 1); }} className={cn(page === 1 ? "pointer-events-none opacity-50" : "")} />
                </PaginationItem>
                {page > 1 && (
                  <PaginationItem>
                    <PaginationLink href="#" onClick={(e) => { e.preventDefault(); handlePageChange(page - 1); }}>{page - 1}</PaginationLink>
                  </PaginationItem>
                )}
                <PaginationItem>
                  <PaginationLink href="#" isActive>{page}</PaginationLink>
                </PaginationItem>
                {hasMore && (
                  <PaginationItem>
                    <PaginationLink href="#" onClick={(e) => { e.preventDefault(); handlePageChange(page + 1); }}>{page + 1}</PaginationLink>
                  </PaginationItem>
                )}
                <PaginationItem>
                  <PaginationNext href="#" onClick={(e) => { e.preventDefault(); if (hasMore) handlePageChange(page + 1); }} className={cn(!hasMore ? "pointer-events-none opacity-50" : "")} />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
