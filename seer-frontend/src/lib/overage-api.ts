import { backendApiClient } from "./api-client";

export interface OverageConfig {
  min_cap_cents: number;
  max_cap_cents: number;
  default_cap_cents: number;
  default_margin_multiplier: number;
  warning_threshold: number;
}

export interface OverageRecordCounts {
  pending: number;
  reported: number;
  failed: number;
}

export interface OverageSettings {
  enabled: boolean;
  eligible: boolean;
  spending_cap_cents: number;
  spending_cap_dollars: number;
  current_usage_cents: number;
  current_usage_dollars: number;
  remaining_cents: number;
  remaining_dollars: number;
  cap_reached: boolean;
  margin_multiplier: number;
  margin_percent: number;
  period_start: string | null;
  enabled_at: string | null;
  records: OverageRecordCounts | null;
}

export interface EnableOverageRequest {
  spending_cap_cents: number;
}

export interface EnableOverageResponse {
  success: boolean;
  settings: OverageSettings;
}

export interface DisableOverageResponse {
  success: boolean;
  pending_charges_cents: number;
  message: string;
}

export interface UpdateCapRequest {
  spending_cap_cents: number;
}

export interface UpdateCapResponse {
  success: boolean;
  spending_cap_cents: number;
  spending_cap_dollars: number;
}

export interface OverageUsageRecord {
  id: number;
  base_cost_cents: number;
  billed_amount_cents: number;
  status: "pending" | "reported" | "failed";
  created_at: string;
  reported_to_stripe_at: string | null;
}

export interface OverageUsageListResponse {
  items: OverageUsageRecord[];
  total_count: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

export const overageApi = {
  async getConfig(): Promise<OverageConfig> {
    return backendApiClient.request<OverageConfig>("/api/overage/config");
  },

  async getSettings(): Promise<OverageSettings> {
    return backendApiClient.request<OverageSettings>("/api/overage");
  },

  async enable(spending_cap_cents: number): Promise<EnableOverageResponse> {
    return backendApiClient.request<EnableOverageResponse>("/api/overage/enable", {
      method: "POST",
      body: { spending_cap_cents },
    });
  },

  async disable(): Promise<DisableOverageResponse> {
    return backendApiClient.request<DisableOverageResponse>("/api/overage/disable", {
      method: "POST",
      body: {},
    });
  },

  async updateCap(spending_cap_cents: number): Promise<UpdateCapResponse> {
    return backendApiClient.request<UpdateCapResponse>("/api/overage/cap", {
      method: "PUT",
      body: { spending_cap_cents },
    });
  },

  async getUsageRecords(
    page: number = 1,
    pageSize: number = 20,
    status?: "pending" | "reported" | "failed"
  ): Promise<OverageUsageListResponse> {
    const params = new URLSearchParams({
      page: String(page),
      page_size: String(pageSize),
    });
    if (status) {
      params.append("status", status);
    }
    return backendApiClient.request<OverageUsageListResponse>(
      `/api/overage/usage?${params.toString()}`
    );
  },
};
