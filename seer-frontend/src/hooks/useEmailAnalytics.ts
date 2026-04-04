import { useQuery, keepPreviousData } from "@tanstack/react-query";
import {
  emailAnalyticsApi,
  type EmailEventsParams,
  type EmailSummaryParams,
} from "@/lib/email-analytics-api";
import { emailAnalyticsKeys } from "@/lib/query-keys";

export function useEmailEvents(params: EmailEventsParams = {}) {
  return useQuery({
    queryKey: emailAnalyticsKeys.events(params),
    queryFn: () => emailAnalyticsApi.getEvents(params),
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  });
}

export function useEmailSummary(params: EmailSummaryParams = {}) {
  return useQuery({
    queryKey: emailAnalyticsKeys.summary(params),
    queryFn: () => emailAnalyticsApi.getSummary(params),
    staleTime: 60_000,
  });
}
