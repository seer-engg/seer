/**
 * Browser API functions for profile management, sessions, and recordings.
 *
 * Wraps REST endpoints using the backendApiClient.
 */
import { backendApiClient, getBackendBaseUrl } from './api-client';
import type {
  BrowserProfile,
  CreateSessionResponse,
  CompleteSessionResponse,
  RecordingListResponse,
  RecordingMetadata,
  RecordingEventsResponse,
} from '@/types/browser';

// ============================================================================
// Browser Profiles
// ============================================================================

/**
 * Create a new browser profile.
 */
export async function createBrowserProfile(name: string): Promise<BrowserProfile> {
  return backendApiClient.request<BrowserProfile>('/api/browser/profiles', {
    method: 'POST',
    body: { name },
  });
}

/**
 * List all browser profiles for the current user.
 */
export async function listBrowserProfiles(): Promise<BrowserProfile[]> {
  return backendApiClient.request<BrowserProfile[]>('/api/browser/profiles');
}

/**
 * Get a specific browser profile by ID.
 */
export async function getBrowserProfile(profileId: string): Promise<BrowserProfile> {
  return backendApiClient.request<BrowserProfile>(`/api/browser/profiles/${profileId}`);
}

/**
 * Delete a browser profile.
 */
export async function deleteBrowserProfile(profileId: string): Promise<{ deleted: boolean; profile_id: string }> {
  return backendApiClient.request<{ deleted: boolean; profile_id: string }>(`/api/browser/profiles/${profileId}`, {
    method: 'DELETE',
  });
}

// ============================================================================
// Browser Sessions (Interactive Streaming)
// ============================================================================

export interface CreateSessionParams {
  profileId: string;
  targetUrl?: string;
}

/**
 * Create an interactive browser session for streaming.
 * Returns a session ID and WebSocket URL.
 */
export async function createBrowserSession(params: CreateSessionParams): Promise<CreateSessionResponse> {
  return backendApiClient.request<CreateSessionResponse>('/api/browser/sessions', {
    method: 'POST',
    body: {
      profile_id: params.profileId,
      target_url: params.targetUrl,
    },
  });
}

/**
 * Complete an interactive session, saving encrypted state.
 */
export async function completeBrowserSession(sessionId: string): Promise<CompleteSessionResponse> {
  return backendApiClient.request<CompleteSessionResponse>(`/api/browser/sessions/${sessionId}/complete`, {
    method: 'POST',
  });
}

// ============================================================================
// Session Recordings
// ============================================================================

export interface ListRecordingsParams {
  profileId?: string;
  workflowRunId?: string;
  limit?: number;
  offset?: number;
}

/**
 * List session recordings for the current user.
 */
export async function listRecordings(params: ListRecordingsParams = {}): Promise<RecordingListResponse> {
  const searchParams = new URLSearchParams();
  if (params.profileId) searchParams.set('profile_id', params.profileId);
  if (params.workflowRunId) searchParams.set('workflow_run_id', params.workflowRunId);
  if (params.limit) searchParams.set('limit', String(params.limit));
  if (params.offset) searchParams.set('offset', String(params.offset));

  const query = searchParams.toString();
  return backendApiClient.request<RecordingListResponse>(
    `/api/browser/recordings${query ? `?${query}` : ''}`
  );
}

/**
 * Get recording metadata (without event data).
 */
export async function getRecording(recordingId: string): Promise<RecordingMetadata> {
  return backendApiClient.request<RecordingMetadata>(`/api/browser/recordings/${recordingId}`);
}

/**
 * Get decompressed rrweb events for a recording.
 */
export async function getRecordingEvents(recordingId: string): Promise<RecordingEventsResponse> {
  return backendApiClient.request<RecordingEventsResponse>(`/api/browser/recordings/${recordingId}/events`);
}

/**
 * Get recording events for public sharing (no auth required).
 */
export async function getSharedRecordingEvents(recordingId: string): Promise<RecordingEventsResponse> {
  const baseUrl = getBackendBaseUrl();
  const res = await fetch(`${baseUrl}/api/browser/recordings/shared/${recordingId}/events`);
  if (!res.ok) {
    throw new Error(`Failed to fetch shared recording: ${res.status}`);
  }
  return res.json();
}

/**
 * Delete a session recording.
 */
export async function deleteRecording(recordingId: string): Promise<{ deleted: boolean; recording_id: string }> {
  return backendApiClient.request<{ deleted: boolean; recording_id: string }>(
    `/api/browser/recordings/${recordingId}`,
    { method: 'DELETE' }
  );
}
