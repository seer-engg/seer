import { backendApiClient } from './api-client';
import type {
  MeetingsListResponse,
  MeetingListItem,
  TranscriptResponse,
  RecordingResponse,
} from '@/types/meetings';

export const meetingsApi = {
  async list(): Promise<MeetingsListResponse> {
    return backendApiClient.request<MeetingsListResponse>('/api/meetings');
  },

  async getDetail(botId: string): Promise<MeetingListItem> {
    return backendApiClient.request<MeetingListItem>(`/api/meetings/${botId}`);
  },

  async getTranscript(botId: string): Promise<TranscriptResponse> {
    return backendApiClient.request<TranscriptResponse>(`/api/meetings/${botId}/transcript`);
  },

  async getRecording(botId: string): Promise<RecordingResponse> {
    return backendApiClient.request<RecordingResponse>(`/api/meetings/${botId}/recording`);
  },
};
