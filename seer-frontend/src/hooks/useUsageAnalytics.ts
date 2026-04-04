import { useQuery, keepPreviousData } from "@tanstack/react-query";
import {
  usageAnalyticsApi,
  type AnalyticsParams,
  type WorkflowAnalyticsParams,
  type RecordsParams,
} from "@/lib/usage-analytics-api";
import { analyticsKeys } from "@/lib/query-keys";

export function useUsageAnalytics(params: AnalyticsParams = {}) {
  return useQuery({
    queryKey: analyticsKeys.summary(params),
    queryFn: () => usageAnalyticsApi.getAnalytics(params),
    staleTime: 60_000,
  });
}

export function useWorkflowAnalytics(params: WorkflowAnalyticsParams = {}) {
  return useQuery({
    queryKey: analyticsKeys.workflows(params),
    queryFn: () => usageAnalyticsApi.getWorkflowAnalytics(params),
    staleTime: 60_000,
  });
}

export function useUsageRecords(params: RecordsParams = {}) {
  return useQuery({
    queryKey: analyticsKeys.records(params),
    queryFn: () => usageAnalyticsApi.getRecords(params),
    staleTime: 30_000,
    placeholderData: keepPreviousData,
  });
}
