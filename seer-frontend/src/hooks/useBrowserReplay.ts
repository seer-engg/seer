/**
 * Hook for rrweb session replay.
 *
 * Wraps useBrowserRecordingEvents with replay-specific state.
 */
import { useMemo } from 'react';
import { useBrowserRecordingEvents, useBrowserRecording } from './useBrowserRecordings';
import type { RecordingMetadata } from '@/types/browser';

export interface UseBrowserReplayReturn {
  /** Recording metadata. */
  metadata: RecordingMetadata | null;
  /** rrweb events for replay. */
  events: unknown[];
  /** Event count. */
  eventCount: number;
  /** Whether data is loading. */
  isLoading: boolean;
  /** Error if fetch failed. */
  error: Error | null;
  /** Whether events are ready for replay. */
  isReady: boolean;
}

export function useBrowserReplay(recordingId: string | null, options?: { public?: boolean }): UseBrowserReplayReturn {
  const isPublic = options?.public ?? false;

  const {
    data: metadata,
    isLoading: metadataLoading,
    error: metadataError,
  } = useBrowserRecording(isPublic ? null : recordingId);

  const {
    data: eventsData,
    isLoading: eventsLoading,
    error: eventsError,
  } = useBrowserRecordingEvents(recordingId, { public: isPublic });

  const isLoading = metadataLoading || eventsLoading;
  const error = (metadataError || eventsError) as Error | null;

  const events = useMemo(() => {
    return eventsData?.events ?? [];
  }, [eventsData]);

  return {
    metadata: metadata ?? null,
    events,
    eventCount: eventsData?.event_count ?? 0,
    isLoading,
    error,
    isReady: !isLoading && !error && events.length > 0,
  };
}
