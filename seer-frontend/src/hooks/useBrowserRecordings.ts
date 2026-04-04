/**
 * Hook for browser session recordings with React Query.
 *
 * Provides list, fetch, and delete operations for recordings.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  listRecordings,
  getRecording,
  getRecordingEvents,
  getSharedRecordingEvents,
  deleteRecording,
} from '@/lib/browser-api';
import { browserKeys } from '@/lib/query-keys';
import type { RecordingMetadata, RecordingEventsResponse } from '@/types/browser';

export interface UseBrowserRecordingsOptions {
  /** Filter by browser profile ID. */
  profileId?: string;
  /** Filter by workflow run ID. */
  workflowRunId?: string;
  /** Number of recordings to fetch. */
  limit?: number;
  /** Offset for pagination. */
  offset?: number;
  /** Whether to enable the query. */
  enabled?: boolean;
}

export interface UseBrowserRecordingsReturn {
  /** List of recordings. */
  recordings: RecordingMetadata[];
  /** Total count of recordings. */
  total: number;
  /** Whether recordings are loading. */
  isLoading: boolean;
  /** Error if recordings failed to load. */
  error: Error | null;
  /** Delete a recording. */
  deleteRecording: (recordingId: string) => Promise<void>;
  /** Whether a delete is in progress. */
  isDeleting: boolean;
  /** Refetch recordings. */
  refetch: () => Promise<void>;
}

export function useBrowserRecordings(
  options: UseBrowserRecordingsOptions = {}
): UseBrowserRecordingsReturn {
  const queryClient = useQueryClient();
  const { profileId, workflowRunId, limit = 20, offset = 0, enabled = true } = options;

  const queryKey = browserKeys.recordingList({
    enabled,
    limit,
    offset,
    profileId,
    workflowRunId,
  });

  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey,
    queryFn: () => listRecordings({ profileId, workflowRunId, limit, offset }),
    enabled,
    staleTime: 30 * 1000, // 30 seconds
  });

  const deleteMutation = useMutation({
    mutationFn: deleteRecording,
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: browserKeys.recordings() });
    },
  });

  return {
    recordings: data?.recordings ?? [],
    total: data?.total ?? 0,
    isLoading,
    error: error as Error | null,
    deleteRecording: async (recordingId: string) => {
      await deleteMutation.mutateAsync(recordingId);
    },
    isDeleting: deleteMutation.isPending,
    refetch: async () => {
      await refetch();
    },
  };
}

/**
 * Hook for fetching a single recording's metadata.
 */
export function useBrowserRecording(recordingId: string | null) {
  return useQuery({
    queryKey: browserKeys.recordingDetail(recordingId),
    queryFn: () => getRecording(recordingId!),
    enabled: !!recordingId,
    staleTime: Infinity, // Recording metadata is immutable
  });
}

/**
 * Hook for fetching recording events for replay.
 */
export function useBrowserRecordingEvents(recordingId: string | null, options?: { public?: boolean }) {
  const isPublic = options?.public ?? false;
  return useQuery<RecordingEventsResponse>({
    queryKey: browserKeys.recordingEvents(recordingId, isPublic),
    queryFn: () => isPublic ? getSharedRecordingEvents(recordingId!) : getRecordingEvents(recordingId!),
    enabled: !!recordingId,
    staleTime: Infinity, // Events are immutable once recorded
  });
}
