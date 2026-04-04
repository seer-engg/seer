/**
 * Browser streaming and recording types.
 *
 * Defines interfaces for WebSocket messaging protocol, API responses,
 * and browser profile/recording metadata.
 */

// ============================================================================
// WebSocket Message Types
// ============================================================================

/** Stream connection status. */
export type StreamStatus = 'connecting' | 'connected' | 'navigating' | 'error' | 'closed';

/** Server → Client: Video frame. */
export interface FrameMessage {
  type: 'frame';
  data: string; // base64-encoded JPEG
  timestamp: number;
}

/** Server → Client: Status update. */
export interface StatusMessage {
  type: 'status';
  status: StreamStatus;
  message?: string;
}

/** Server → Client: Error notification. */
export interface ErrorMessage {
  type: 'error';
  code: string;
  message: string;
}

/** Union of all server messages. */
export type ServerMessage = FrameMessage | StatusMessage | ErrorMessage;

// ============================================================================
// Client → Server Messages
// ============================================================================

/** Mouse event. */
export interface MouseMessage {
  type: 'mouse';
  action: 'click' | 'move' | 'down' | 'up';
  x: number;
  y: number;
  button?: 'left' | 'middle' | 'right';
  clickCount?: number;
}

/** Keyboard event. */
export interface KeyMessage {
  type: 'key';
  action: 'down' | 'up' | 'press';
  key: string;
  code: string;
  text?: string;
}

/** Scroll event. */
export interface ScrollMessage {
  type: 'scroll';
  x: number;
  y: number;
  deltaX: number;
  deltaY: number;
}

/** Navigation request. */
export interface NavigateMessage {
  type: 'navigate';
  url: string;
}

/** Union of all client messages. */
export type ClientMessage = MouseMessage | KeyMessage | ScrollMessage | NavigateMessage;

// ============================================================================
// API Response Types
// ============================================================================

/** Browser profile metadata. */
export interface BrowserProfile {
  id: string;
  name: string;
  logged_in_domains: string[];
  created_at: string | null;
  last_used_at: string | null;
}

/** Response from POST /api/browser/sessions. */
export interface CreateSessionResponse {
  session_id: string;
  profile_id: string;
  ws_url: string;
  recording_id: string | null;
  status: string;
}

/** Response from POST /api/browser/sessions/{id}/complete. */
export interface CompleteSessionResponse {
  profile_id: string;
  logged_in_domains: string[];
  recording_id: string | null;
  status: string;
}

/** Recording metadata. */
export interface RecordingMetadata {
  id: string;
  session_type: string;
  event_count: number;
  duration_ms: number;
  compressed_size_bytes: number;
  start_url: string | null;
  status: string;
  created_at: string | null;
  completed_at: string | null;
  browser_profile_id: string | null;
  workflow_run_id: string | null;
}

/** Response from GET /api/browser/recordings. */
export interface RecordingListResponse {
  recordings: RecordingMetadata[];
  total: number;
}

/** Response from GET /api/browser/recordings/{id}/events. */
export interface RecordingEventsResponse {
  recording_id: string;
  event_count: number;
  events: unknown[]; // rrweb event format
}
