import { useEffect } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { type InvoiceItem, type PaymentItem } from "@/lib/subscription-api";
import { cn } from "@/lib/utils";
import { useBillingHistoryStore } from "@/stores/billingHistoryStore";
import { ExternalLink, FileDown, Receipt } from "lucide-react";

const formatDateTime = (isoDate?: string | null): string => {
  if (!isoDate) return "—";
  const date = new Date(isoDate);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
};

const formatCurrencyAmount = (amount?: number | null, currency?: string | null): string => {
  if (amount === null || amount === undefined) return "—";
  const normalizedCurrency = currency?.toUpperCase() || "USD";
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: normalizedCurrency,
      minimumFractionDigits: 2,
    }).format(amount / 100);
  } catch {
    return `${amount / 100} ${normalizedCurrency}`;
  }
};

function StatusBadge({ status }: { status?: string | null }) {
  if (!status) return <Badge variant="secondary">unknown</Badge>;
  const normalized = status.toLowerCase();
  const styleMap: Record<string, string> = {
    paid: "bg-success/15 text-success border-success/40",
    succeeded: "bg-success/15 text-success border-success/40",
    open: "bg-primary/10 text-primary border-primary/40",
    draft: "bg-muted text-foreground border-muted",
    void: "bg-muted text-foreground border-muted",
    uncollectible: "bg-destructive/15 text-destructive border-destructive/30",
    failed: "bg-destructive/15 text-destructive border-destructive/30",
    processing: "bg-amber-500/15 text-amber-700 border-amber-500/30",
  };
  return (
    <Badge variant="outline" className={cn("capitalize", styleMap[normalized] ?? "bg-muted/70")}>
      {normalized.replace("_", " ")}
    </Badge>
  );
}

function HistoryPagination({
  page,
  hasMore,
  onPageChange,
  disabled,
}: {
  page: number;
  hasMore: boolean;
  onPageChange: (next: number) => void;
  disabled?: boolean;
}) {
  const prevPage = Math.max(1, page - 1);
  const nextPage = page + 1;

  return (
    <Pagination className="pt-4">
      <PaginationContent>
        <PaginationItem>
          <PaginationPrevious
            href="#"
            onClick={(event) => {
              event.preventDefault();
              if (page > 1 && !disabled) onPageChange(prevPage);
            }}
            className={cn(page === 1 || disabled ? "pointer-events-none opacity-50" : "")}
          />
        </PaginationItem>
        {page > 1 ? (
          <PaginationItem>
            <PaginationLink
              href="#"
              onClick={(event) => {
                event.preventDefault();
                if (!disabled) onPageChange(prevPage);
              }}
            >
              {prevPage}
            </PaginationLink>
          </PaginationItem>
        ) : null}
        <PaginationItem>
          <PaginationLink href="#" isActive>
            {page}
          </PaginationLink>
        </PaginationItem>
        {hasMore ? (
          <PaginationItem>
            <PaginationLink
              href="#"
              onClick={(event) => {
                event.preventDefault();
                if (!disabled) onPageChange(nextPage);
              }}
            >
              {nextPage}
            </PaginationLink>
          </PaginationItem>
        ) : null}
        <PaginationItem>
          <PaginationNext
            href="#"
            onClick={(event) => {
              event.preventDefault();
              if (hasMore && !disabled) onPageChange(nextPage);
            }}
            className={cn(!hasMore || disabled ? "pointer-events-none opacity-50" : "")}
          />
        </PaginationItem>
      </PaginationContent>
    </Pagination>
  );
}

interface InvoiceTableProps {
  items: InvoiceItem[] | null;
  paginationPage: number;
  hasMore: boolean;
  isLoading: boolean;
  error: string | null;
  onPageChange: (page: number) => void;
}

function InvoiceTable({ items, paginationPage, hasMore, isLoading, error, onPageChange }: InvoiceTableProps) {
  const dataReady = items && !isLoading;
  const hasRows = Boolean(items?.length);
  const downloadLabel = "Download";

  return (
    <div className="space-y-3">
      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Unable to load invoices</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}
      <div className="overflow-hidden rounded-lg border border-border/60">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Invoice</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Date</TableHead>
              <TableHead className="text-right">Action</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && !hasRows
              ? Array.from({ length: 4 }).map((_, idx) => (
                  <TableRow key={`invoice-skeleton-${idx}`}>
                    <TableCell>
                      <Skeleton className="h-4 w-28" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-16" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-20" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-24" />
                    </TableCell>
                    <TableCell className="text-right">
                      <Skeleton className="ml-auto h-8 w-24" />
                    </TableCell>
                  </TableRow>
                ))
              : null}
            {dataReady && hasRows
              ? items!.map((invoice) => {
                  const url = invoice.invoice_pdf || invoice.hosted_invoice_url || undefined;
                  return (
                    <TableRow key={invoice.id}>
                      <TableCell>
                        <div className="flex flex-col">
                          <span className="font-medium">#{invoice.number ?? invoice.id}</span>
                          <span className="text-xs text-muted-foreground">
                            {invoice.billing_reason ? invoice.billing_reason.replace("_", " ") : "Subscription"}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={invoice.status} />
                      </TableCell>
                      <TableCell>{formatCurrencyAmount(invoice.total, invoice.currency)}</TableCell>
                      <TableCell>{formatDateTime(invoice.created_at)}</TableCell>
                      <TableCell className="text-right">
                        <Button
                          asChild
                          size="sm"
                          variant="outline"
                          disabled={!url}
                          className="inline-flex items-center gap-1"
                        >
                          <a href={url ?? "#"} target="_blank" rel="noopener noreferrer">
                            <FileDown className="h-4 w-4" />
                            {downloadLabel}
                          </a>
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })
              : null}
            {dataReady && !hasRows ? (
              <TableRow>
                <TableCell colSpan={5} className="py-10 text-center text-muted-foreground">
                  No invoices yet.
                </TableCell>
              </TableRow>
            ) : null}
          </TableBody>
        </Table>
      </div>
      <HistoryPagination
        page={paginationPage}
        hasMore={hasMore}
        onPageChange={onPageChange}
        disabled={isLoading}
      />
    </div>
  );
}

interface PaymentTableProps {
  items: PaymentItem[] | null;
  paginationPage: number;
  hasMore: boolean;
  isLoading: boolean;
  error: string | null;
  onPageChange: (page: number) => void;
}

function PaymentTable({ items, paginationPage, hasMore, isLoading, error, onPageChange }: PaymentTableProps) {
  const dataReady = items && !isLoading;
  const hasRows = Boolean(items?.length);

  return (
    <div className="space-y-3">
      {error ? (
        <Alert variant="destructive">
          <AlertTitle>Unable to load payments</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      ) : null}
      <div className="overflow-hidden rounded-lg border border-border/60">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Payment</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Amount</TableHead>
              <TableHead>Date</TableHead>
              <TableHead className="text-right">Receipt</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading && !hasRows
              ? Array.from({ length: 4 }).map((_, idx) => (
                  <TableRow key={`payment-skeleton-${idx}`}>
                    <TableCell>
                      <Skeleton className="h-4 w-24" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-16" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-20" />
                    </TableCell>
                    <TableCell>
                      <Skeleton className="h-4 w-24" />
                    </TableCell>
                    <TableCell className="text-right">
                      <Skeleton className="ml-auto h-8 w-24" />
                    </TableCell>
                  </TableRow>
                ))
              : null}
            {dataReady && hasRows
              ? items!.map((payment) => (
                  <TableRow key={payment.id}>
                    <TableCell>
                      <div className="flex flex-col">
                        <span className="font-medium">{payment.description || "Payment"}</span>
                        <span className="text-xs text-muted-foreground">{payment.id}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <StatusBadge status={payment.status} />
                    </TableCell>
                    <TableCell>{formatCurrencyAmount(payment.amount, payment.currency)}</TableCell>
                    <TableCell>{formatDateTime(payment.created_at)}</TableCell>
                    <TableCell className="text-right">
                      <Button
                        asChild
                        size="sm"
                        variant="outline"
                        disabled={!payment.receipt_url}
                        className="inline-flex items-center gap-1"
                      >
                        <a href={payment.receipt_url ?? "#"} target="_blank" rel="noopener noreferrer">
                          <Receipt className="h-4 w-4" />
                          View receipt
                        </a>
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              : null}
            {dataReady && !hasRows ? (
              <TableRow>
                <TableCell colSpan={5} className="py-10 text-center text-muted-foreground">
                  No payments found.
                </TableCell>
              </TableRow>
            ) : null}
          </TableBody>
        </Table>
      </div>
      <HistoryPagination
        page={paginationPage}
        hasMore={hasMore}
        onPageChange={onPageChange}
        disabled={isLoading}
      />
    </div>
  );
}

export function BillingHistorySection() {
  const invoices = useBillingHistoryStore((state) => state.invoices);
  const payments = useBillingHistoryStore((state) => state.payments);
  const invoicePagination = useBillingHistoryStore((state) => state.invoicePagination);
  const paymentPagination = useBillingHistoryStore((state) => state.paymentPagination);
  const invoiceError = useBillingHistoryStore((state) => state.invoiceError);
  const paymentError = useBillingHistoryStore((state) => state.paymentError);
  const isLoadingInvoices = useBillingHistoryStore((state) => state.isLoadingInvoices);
  const isLoadingPayments = useBillingHistoryStore((state) => state.isLoadingPayments);
  const fetchInvoices = useBillingHistoryStore((state) => state.fetchInvoices);
  const fetchPayments = useBillingHistoryStore((state) => state.fetchPayments);

  useEffect(() => {
    fetchInvoices(1);
    fetchPayments(1);
  }, [fetchInvoices, fetchPayments]);

  return (
    <Card>
      <CardHeader className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <CardTitle className="text-lg">Billing history</CardTitle>
          <CardDescription>Invoices and payment receipts directly from Stripe.</CardDescription>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <ExternalLink className="h-4 w-4" />
          Links open in a new tab.
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <Tabs defaultValue="invoices" className="w-full">
          <TabsList className="w-full sm:w-auto">
            <TabsTrigger value="invoices" className="flex-1 sm:flex-none">
              Invoices
            </TabsTrigger>
            <TabsTrigger value="payments" className="flex-1 sm:flex-none">
              Payments
            </TabsTrigger>
          </TabsList>
          <TabsContent value="invoices" className="mt-4">
            <InvoiceTable
              items={invoices}
              paginationPage={invoicePagination.page}
              hasMore={invoicePagination.hasMore}
              isLoading={isLoadingInvoices}
              error={invoiceError}
              onPageChange={fetchInvoices}
            />
          </TabsContent>
          <TabsContent value="payments" className="mt-4">
            <PaymentTable
              items={payments}
              paginationPage={paymentPagination.page}
              hasMore={paymentPagination.hasMore}
              isLoading={isLoadingPayments}
              error={paymentError}
              onPageChange={fetchPayments}
            />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
