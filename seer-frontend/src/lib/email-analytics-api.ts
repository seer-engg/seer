import { backendApiClient } from "./api-client";

export interface EmailEventsParams {
  limit?: number;
  offset?: number;
  email_type?: string;
  event_type?: string;
  start?: string;
  end?: string;
}

export interface EmailSummaryParams {
  start?: string;
  end?: string;
}

export interface EmailEventItem {
  id: string;
  tracking_id: string;
  provider: string;
  email_type: string;
  event_type: string;
  recipient: string;
  subject: string | null;
  link_url: string | null;
  workflow_run_id: string | null;
  created_at: string;
}

export interface EmailEventsResponse {
  events: EmailEventItem[];
  total: number;
}

export interface EmailSummaryResponse {
  total_sent: number;
  total_opened: number;
  total_clicked: number;
  open_rate: number;
  click_rate: number;
}

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

export const emailAnalyticsApi = {
  async getEvents(params: EmailEventsParams = {}): Promise<EmailEventsResponse> {
    const qs = toQueryString([
      ["limit", params.limit],
      ["offset", params.offset],
      ["email_type", params.email_type],
      ["event_type", params.event_type],
      ["start", params.start],
      ["end", params.end],
    ]);
    return backendApiClient.request<EmailEventsResponse>(`/api/email-analytics/events${qs}`);
  },

  async getSummary(params: EmailSummaryParams = {}): Promise<EmailSummaryResponse> {
    const qs = toQueryString([
      ["start", params.start],
      ["end", params.end],
    ]);
    return backendApiClient.request<EmailSummaryResponse>(`/api/email-analytics/summary${qs}`);
  },
};
