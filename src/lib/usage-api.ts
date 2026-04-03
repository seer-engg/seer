import { backendApiClient } from "./api-client";

export interface UsageMetric {
  used: number;
  limit: number | null;
  remaining: number | null;
  is_unlimited: boolean;
  reset_at?: string | null;
  disabled?: boolean;
  unit?: string | null;
}

export interface UsageSummary {
  is_self_hosted: boolean;
  limits: {
    poll_min_interval_seconds: number;
  };
  usage: {
    workflow_runs: UsageMetric;
    llm_credits: UsageMetric;
    llm_credits_5h: UsageMetric;
    llm_credits_weekly: UsageMetric;
    workflows: UsageMetric;
  };
}

export const usageApi = {
  async getUsageSummary(): Promise<UsageSummary> {
    return backendApiClient.request<UsageSummary>("/api/usage");
  },
};
