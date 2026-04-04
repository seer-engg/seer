import { useCallback, useEffect, useRef } from 'react';
import { useGeneralChatStore } from '@/stores/generalChatStore';
import {
  createChatSession,
  listChatSessions,
  getChatSession,
  deleteChatSession,
  sendChatMessage,
  pollChatStatus,
  renameChatSession,
  streamChatMessage,
} from '@/lib/general-chat-api';

const POLL_INTERVAL = 1500;

export function useGeneralChat() {
  const pollingEnabled = useGeneralChatStore((s) => s.pollingEnabled);
  const activeSessionId = useGeneralChatStore((s) => s.activeSessionId);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const loadSessions = useCallback(async () => {
    const sessions = await listChatSessions();
    useGeneralChatStore.getState().setSessions(sessions);
  }, []);

  const selectSession = useCallback(async (sessionId: number) => {
    const state = useGeneralChatStore.getState();
    state.setActiveSessionId(sessionId);
    const detail = await getChatSession(sessionId);
    state.setMessages(detail.messages);
    if (detail.current_execution_status === 'running' || detail.current_execution_status === 'queued') {
      state.startPolling();
    }
  }, []);

  const createSession = useCallback(async () => {
    const session = await createChatSession();
    const state = useGeneralChatStore.getState();
    state.setSessions([session, ...state.sessions]);
    state.setActiveSessionId(session.id);
    state.setMessages([]);
    return session;
  }, []);

  const removeSession = useCallback(async (sessionId: number) => {
    await deleteChatSession(sessionId);
    const state = useGeneralChatStore.getState();
    state.setSessions(state.sessions.filter((s) => s.id !== sessionId));
    if (state.activeSessionId === sessionId) {
      state.setActiveSessionId(null);
      state.setMessages([]);
    }
  }, []);

  const renameSession = useCallback(async (sessionId: number, title: string) => {
    const updated = await renameChatSession(sessionId, title);
    const state = useGeneralChatStore.getState();
    state.setSessions(state.sessions.map((s) => (s.id === sessionId ? { ...s, title: updated.title } : s)));
  }, []);

  const handleSend = useCallback(async () => {
    const state = useGeneralChatStore.getState();
    const rawInput = state.input.trim();
    if (!rawInput || state.isLoading) return;

    // Prepend attached file contents
    let input = '';
    for (const file of state.attachedFiles) {
      input += `[Attached: ${file.name}]\n${file.content}\n[End: ${file.name}]\n\n`;
    }
    input += rawInput;

    let sessionId = state.activeSessionId;
    if (!sessionId) {
      const session = await createChatSession();
      sessionId = session.id;
      state.setSessions([session, ...state.sessions]);
      state.setActiveSessionId(sessionId);
    }

    state.addMessage({
      id: Date.now(),
      role: 'user',
      content: input,
      created_at: new Date().toISOString(),
    });
    state.setInput('');
    state.clearAttachedFiles();
    state.setIsLoading(true);

    const msgOptions = {
      model: state.selectedModel || undefined,
      generate_image: state.generateImage,
      ...(state.generateImage && state.selectedImageModel ? { image_model: state.selectedImageModel } : {}),
    };

    if (state.generateImage) {
      // Image generation: use existing polling path
      try {
        await sendChatMessage(sessionId, input, msgOptions);
        state.startPolling();
      } catch (error) {
        console.error('Failed to send message:', error);
        state.setIsLoading(false);
      }
    } else {
      // Text chat: stream via SSE
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      useGeneralChatStore.setState({ isStreaming: true, streamingContent: '' });

      try {
        let fullContent = '';
        await streamChatMessage(sessionId, input, msgOptions, (token) => {
          fullContent += token;
          state.appendStreamingContent(token);
        }, controller.signal);
        state.finalizeStreaming(fullContent);
        loadSessions();
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          console.error('Streaming failed:', error);
          useGeneralChatStore.setState({ isStreaming: false, streamingContent: '', isLoading: false });
          state.addMessage({
            id: Date.now(),
            role: 'assistant',
            content: (error as Error).message || 'An error occurred.',
            created_at: new Date().toISOString(),
          });
        }
      }
    }
  }, [loadSessions]);

  // Polling effect
  useEffect(() => {
    if (!pollingEnabled || !activeSessionId) {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    const poll = async () => {
      const currentState = useGeneralChatStore.getState();
      if (!currentState.pollingEnabled || !currentState.activeSessionId) return;

      try {
        const status = await pollChatStatus(currentState.activeSessionId);
        currentState.setExecutionStatus(status.status);

        if (status.status === 'completed' || status.status === 'failed') {
          currentState.stopPolling();
          currentState.setIsLoading(false);

          if (status.status === 'completed' && status.response) {
            currentState.addMessage({
              id: Date.now(),
              role: 'assistant',
              content: status.response,
              image_urls: status.image_urls ?? undefined,
              thinking: status.thinking ?? undefined,
              created_at: new Date().toISOString(),
            });
            loadSessions();
          } else if (status.status === 'failed') {
            currentState.addMessage({
              id: Date.now(),
              role: 'assistant',
              content: (status.error?.detail as string) || 'An error occurred.',
              created_at: new Date().toISOString(),
            });
          }
        }
      } catch (error) {
        console.error('Polling error:', error);
        const s = useGeneralChatStore.getState();
        s.stopPolling();
        s.setIsLoading(false);
      }
    };

    poll();
    pollTimerRef.current = setInterval(poll, POLL_INTERVAL);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [pollingEnabled, activeSessionId, loadSessions]);

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    const state = useGeneralChatStore.getState();
    if (state.pollingEnabled) {
      state.stopPolling();
    }
    if (state.isStreaming && state.streamingContent) {
      state.finalizeStreaming(state.streamingContent);
    } else {
      useGeneralChatStore.setState({ isStreaming: false, streamingContent: '' });
    }
    state.setIsLoading(false);
  }, []);

  return { loadSessions, selectSession, createSession, removeSession, renameSession, handleSend, handleStop };
}
