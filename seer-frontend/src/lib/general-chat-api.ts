import { backendApiClient, backendTokenProvider, getBackendBaseUrl } from './api-client';

export interface ChatSessionResponse {
  id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  current_execution_status: string | null;
}

export interface ChatMessageResponse {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  image_urls?: string[];
  thinking?: string[];
  created_at: string;
}

export interface ChatSessionDetailResponse extends ChatSessionResponse {
  messages: ChatMessageResponse[];
}

export interface ChatStatusResponse {
  status: string | null;
  response: string | null;
  image_urls: string[] | null;
  thinking: string[] | null;
  error: Record<string, unknown> | null;
}

export interface ChatSendResponse {
  session_id: number;
  execution_status: string;
}

export async function createChatSession(title?: string): Promise<ChatSessionResponse> {
  return backendApiClient.request('/api/chat/sessions', {
    method: 'POST',
    body: { title },
  });
}

export async function listChatSessions(limit = 50, offset = 0): Promise<ChatSessionResponse[]> {
  return backendApiClient.request(`/api/chat/sessions?limit=${limit}&offset=${offset}`);
}

export async function getChatSession(sessionId: number): Promise<ChatSessionDetailResponse> {
  return backendApiClient.request(`/api/chat/sessions/${sessionId}`);
}

export async function deleteChatSession(sessionId: number): Promise<void> {
  return backendApiClient.request(`/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
}

export async function renameChatSession(sessionId: number, title: string): Promise<ChatSessionResponse> {
  return backendApiClient.request(`/api/chat/sessions/${sessionId}`, {
    method: 'PATCH',
    body: { title },
  });
}

export async function sendChatMessage(
  sessionId: number,
  message: string,
  options?: { model?: string; generate_image?: boolean; image_model?: string; image_size?: string }
): Promise<ChatSendResponse> {
  return backendApiClient.request(`/api/chat/sessions/${sessionId}/messages`, {
    method: 'POST',
    body: { message, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone, ...options },
  });
}

export async function pollChatStatus(sessionId: number): Promise<ChatStatusResponse> {
  return backendApiClient.request(`/api/chat/sessions/${sessionId}/status`);
}

export async function streamChatMessage(
  sessionId: number,
  message: string,
  options: { model?: string; generate_image?: boolean; image_model?: string; image_size?: string } | undefined,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  const token = await backendTokenProvider();
  const baseUrl = getBackendBaseUrl();
  const response = await fetch(`${baseUrl}/api/chat/sessions/${sessionId}/messages/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone, ...options }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = JSON.parse(line.slice(6));
      if (data.error) throw new Error(data.error);
      if (data.done) return;
      if (data.token) onToken(data.token);
    }
  }
}
