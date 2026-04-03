import {
  subscriptionApi,
  type InvoiceItem,
  type PaymentItem,
  type PaginationMeta,
} from "@/lib/subscription-api";
import { createStore } from "./createStore";

interface PaginationState {
  page: number;
  hasMore: boolean;
}

const DEFAULT_PAGE_SIZE = 10;

export interface BillingHistoryState {
  invoices: InvoiceItem[] | null;
  payments: PaymentItem[] | null;
  invoicePagination: PaginationState;
  paymentPagination: PaginationState;
  pageSize: number;
  isLoadingInvoices: boolean;
  isLoadingPayments: boolean;
  invoiceError: string | null;
  paymentError: string | null;
  fetchInvoices: (page?: number) => Promise<void>;
  fetchPayments: (page?: number) => Promise<void>;
}

const toPaginationState = (pagination: PaginationMeta): PaginationState => ({
  page: pagination.page,
  hasMore: pagination.has_more,
});

export const useBillingHistoryStore = createStore<BillingHistoryState>((set, get) => ({
  invoices: null,
  payments: null,
  invoicePagination: { page: 1, hasMore: false },
  paymentPagination: { page: 1, hasMore: false },
  pageSize: DEFAULT_PAGE_SIZE,
  isLoadingInvoices: false,
  isLoadingPayments: false,
  invoiceError: null,
  paymentError: null,

  fetchInvoices: async (page) => {
    const targetPage = page ?? get().invoicePagination.page;
    const { pageSize } = get();
    set({ isLoadingInvoices: true, invoiceError: null });
    try {
      const result = await subscriptionApi.getInvoices(targetPage, pageSize);
      set({
        invoices: result.items,
        invoicePagination: toPaginationState(result.pagination),
        isLoadingInvoices: false,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to load invoices";
      set({ invoiceError: message, isLoadingInvoices: false });
    }
  },

  fetchPayments: async (page) => {
    const targetPage = page ?? get().paymentPagination.page;
    const { pageSize } = get();
    set({ isLoadingPayments: true, paymentError: null });
    try {
      const result = await subscriptionApi.getPayments(targetPage, pageSize);
      set({
        payments: result.items,
        paymentPagination: toPaginationState(result.pagination),
        isLoadingPayments: false,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unable to load payments";
      set({ paymentError: message, isLoadingPayments: false });
    }
  },
}));
