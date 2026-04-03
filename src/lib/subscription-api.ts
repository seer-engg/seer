import { backendApiClient } from "./api-client";

export type SubscriptionTier = "free" | "pro" | "pro_plus" | "ultra";
export type SubscriptionStatus = "active" | "canceled" | "past_due" | "trialing" | "incomplete";

export interface SubscriptionInfo {
  tier: SubscriptionTier;
  status: SubscriptionStatus;
  current_period_end: string | null;
  cancel_at_period_end: boolean;
}

export interface PriceAmount {
  price: number;
  price_id?: string | null;
  original_price?: number | null;
  trial_period_days?: number | null;
  lookup_key?: string | null;
}

export interface PriceTier {
  tier: SubscriptionTier;
  name: string;
  monthly: PriceAmount;
  annual: PriceAmount;
  features?: string[];
  badge?: string | null;
  sort_order?: number;
  description?: string | null;
  upgrade_benefits?: string[];
}

export interface PricingResponse {
  prices: PriceTier[];
}

export interface PaginationMeta {
  page: number;
  page_size: number;
  has_more: boolean;
}

export interface InvoiceItem {
  id: string;
  number?: string | null;
  status?: string | null;
  currency?: string | null;
  total?: number | null;
  amount_paid?: number | null;
  amount_due?: number | null;
  created_at?: string | null;
  period_start?: string | null;
  period_end?: string | null;
  hosted_invoice_url?: string | null;
  invoice_pdf?: string | null;
  billing_reason?: string | null;
}

export interface PaymentItem {
  id: string;
  status?: string | null;
  currency?: string | null;
  amount?: number | null;
  paid?: boolean | null;
  description?: string | null;
  receipt_url?: string | null;
  created_at?: string | null;
  invoice_id?: string | null;
  payment_intent_id?: string | null;
}

export interface InvoiceListResponse {
  items: InvoiceItem[];
  pagination: PaginationMeta;
}

export interface PaymentListResponse {
  items: PaymentItem[];
  pagination: PaginationMeta;
}

export interface CreateSubscriptionWithTrialResponse {
  subscription_id: string;
  status: string;
  trial_end: string | null;
}

export const subscriptionApi = {
  async getCurrentSubscription(): Promise<SubscriptionInfo> {
    return backendApiClient.request<SubscriptionInfo>("/api/subscriptions/current");
  },

  async getPricing(): Promise<PricingResponse> {
    return backendApiClient.request<PricingResponse>("/api/subscriptions/pricing");
  },

  async createCheckout(tier: string, interval: string): Promise<{ checkout_url: string }> {
    return backendApiClient.request<{ checkout_url: string }>("/api/subscriptions/checkout", {
      method: "POST",
      body: { tier, interval },
    });
  },

  async createPortalSession(): Promise<{ portal_url: string }> {
    return backendApiClient.request<{ portal_url: string }>("/api/subscriptions/portal", {
      method: "POST",
      body: {},
    });
  },

  async getInvoices(page: number, pageSize: number): Promise<InvoiceListResponse> {
    const search = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    return backendApiClient.request<InvoiceListResponse>(`/api/subscriptions/invoices?${search.toString()}`);
  },

  async getPayments(page: number, pageSize: number): Promise<PaymentListResponse> {
    const search = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    return backendApiClient.request<PaymentListResponse>(`/api/subscriptions/payments?${search.toString()}`);
  },

  async createSubscriptionWithTrial(
    tier: string,
    interval: string
  ): Promise<CreateSubscriptionWithTrialResponse> {
    return backendApiClient.request<CreateSubscriptionWithTrialResponse>(
      "/api/subscriptions/create-with-trial",
      {
        method: "POST",
        body: { tier, interval },
      }
    );
  },
};
