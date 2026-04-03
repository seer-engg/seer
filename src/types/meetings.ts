export interface MeetingListItem {
  bot_id: string;
  state: string;
  meeting_url: string | null;
  join_at: string | null;
  transcription_state: string | null;
  recording_state: string | null;
  created_at: string | null;
}

export interface MeetingsListResponse {
  items: MeetingListItem[];
}

export interface TranscriptUtterance {
  speaker: string;
  text: string;
  start_ms: number;
  duration_ms: number;
}

export interface TranscriptResponse {
  bot_id: string;
  utterances: TranscriptUtterance[];
}

export interface RecordingResponse {
  url: string | null;
  start_timestamp_ms: number | null;
}
