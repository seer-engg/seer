import { backendApiClient } from "./api-client";

// --- Shared parameter types ---

export interface AnalyticsParams {
  start?: string;
  end?: string;
  model?: string;
  operation?: string;
}

export interface WorkflowAnalyticsParams {
  start?: string;
  end?: string;
}

export interface RecordsParams {
  start?: string;
  end?: string;
  limit?: number;
  offset?: number;
}

// --- Response types for GET /api/usage/analytics ---

export interface DailyTrendItem {
  date: string;
  total_cost: number;
  total_tokens: number;
  call_count: number;
}

export interface ModelBreakdownItem {
  model: string;
  provider: string;
  total_cost: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  call_count: number;
}

export interface OperationBreakdownItem {
  operation: string;
  total_cost: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  call_count: number;
}

export interface UsageAnalyticsResponse {
  total_cost: number;
  total_tokens: number;
  total_calls: number;
  period_start: string;
  period_end: string;
  daily_trend: DailyTrendItem[];
  by_model: ModelBreakdownItem[];
  by_operation: OperationBreakdownItem[];
}

// --- Response types for GET /api/usage/analytics/workflows ---

export interface WorkflowCostItem {
  workflow_id: string;
  workflow_name: string | null;
  total_cost: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_tokens: number;
  call_count: number;
}

export interface WorkflowAnalyticsResponse {
  workflows: WorkflowCostItem[];
  period_start: string;
  period_end: string;
}

// --- Response types for GET /api/usage/analytics/records ---

export interface UsageRecord {
  id: string;
  created_at: string;
  provider: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cost: number;
  operation: string;
  workflow_id: string | null;
  workflow_name: string | null;
}

export interface UsageRecordsResponse {
  records: UsageRecord[];
  total: number;
  limit: number;
  offset: number;
}

// --- API functions ---

function toQueryString(entries: [string, string | number | undefined][]): string {
  const searchParams = new URLSearchParams();
  for (const [key, value] of entries) {
    if (value !== undefined && value !== "") {
      searchParams.set(key, String(value));
    }
  }
  const qs = searchParams.toString();
  return qs ? `?${qs}` : "";
}

export const usageAnalyticsApi = {
  async getAnalytics(params: AnalyticsParams = {}): Promise<UsageAnalyticsResponse> {
    const qs = toQueryString([
      ["start", params.start],
      ["end", params.end],
      ["model", params.model],
      ["operation", params.operation],
    ]);
    return backendApiClient.request<UsageAnalyticsResponse>(`/api/usage/analytics${qs}`);
  },

  async getWorkflowAnalytics(params: WorkflowAnalyticsParams = {}): Promise<WorkflowAnalyticsResponse> {
    const qs = toQueryString([
      ["start", params.start],
      ["end", params.end],
    ]);
    return backendApiClient.request<WorkflowAnalyticsResponse>(`/api/usage/analytics/workflows${qs}`);
  },

  async getRecords(params: RecordsParams = {}): Promise<UsageRecordsResponse> {
    const qs = toQueryString([
      ["start", params.start],
      ["end", params.end],
      ["limit", params.limit],
      ["offset", params.offset],
    ]);
    return backendApiClient.request<UsageRecordsResponse>(`/api/usage/analytics/records${qs}`);
  },
};
