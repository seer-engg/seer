import { useCallback, useEffect, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import type { Node, Edge } from '@xyflow/react';

import type { ChatMessage, ChatSession, WorkflowProposal } from '@/components/workflows/buildtypes';
import {
  createAssistantMessage,
  createErrorMessage,
  handleProposalResponse,
  prepareWorkflowState,
  updateSessionIds,
} from '@/components/workflows/panels/buildAndChatPanelHelpers';
import { backendApiClient, resumeWorkflowChat } from '@/lib/api-client';
import { chatKeys } from '@/lib/query-keys';
import { readSSEStream } from '@/lib/sse-utils';
import { useChatStore, useUIStore } from '@/stores';
import type { ClarificationAnswer, ClarificationAnswers } from '@/types/discovery';
import type { JsonObject } from '@/types/workflow-spec';

// ---------------------------------------------------------------------------
// localStorage helpers — persist session state across page refreshes
// ---------------------------------------------------------------------------

const CHAT_STORAGE_KEY_PREFIX = 'seer_chat_';

interface ActiveStreamState {
  sessionId: number | null;
  threadId: string | null;
  lastEventId: string | null;
  /** 'running' | 'queued' while streaming; null after a terminal event */
  executionStatus: string | null;
}

function persistChatSession(workflowId: string, data: ActiveStreamState & { sessionId: number }): void {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY_PREFIX + workflowId, JSON.stringify(data));
  } catch {
    // localStorage may be unavailable (private browsing, quota exceeded)
  }
}

function restoreChatSession(workflowId: string): (ActiveStreamState & { sessionId: number }) | null {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY_PREFIX + workflowId);
    return raw ? JSON.parse(raw) as ActiveStreamState & { sessionId: number } : null;
  } catch {
    return null;
  }
}

function clearPersistedSession(workflowId: string): void {
  try {
    localStorage.removeItem(CHAT_STORAGE_KEY_PREFIX + workflowId);
  } catch {
    // ignore
  }
}

function persistSelectedSession(workflowId: string, sessionId: number, threadId: string | null): void {
  persistChatSession(workflowId, {
    sessionId,
    threadId,
    lastEventId: null,
    executionStatus: null,
  });
}

function createRequestId(): string {
  return typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function patchStreamRef(
  ref: { current: ActiveStreamState },
  patch: Partial<ActiveStreamState>,
  workflowId: string | null,
): void {
  Object.assign(ref.current, patch);
  const { sessionId, threadId, lastEventId, executionStatus } = ref.current;
  if (workflowId != null && sessionId != null) {
    persistChatSession(workflowId, { sessionId, threadId, lastEventId, executionStatus });
  }
}

function finalizeStreamSession(
  ref: { current: ActiveStreamState },
  workflowId: string | null,
): void {
  if (workflowId && ref.current.sessionId != null) {
    persistSelectedSession(workflowId, ref.current.sessionId, ref.current.threadId);
  }
  ref.current = { sessionId: null, threadId: null, lastEventId: null, executionStatus: null };
}

// eslint-disable-next-line max-lines-per-function
export function useChatActions(workflowId: string | null, nodes: Node[], edges: Edge[]) {
  const queryClient = useQueryClient();

  const input = useChatStore((state) => state.input);
  const isLoading = useChatStore((state) => state.isLoading);
  const selectedModel = useChatStore((state) => state.selectedModel);
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const currentThreadId = useChatStore((state) => state.currentThreadId);
  const pendingInterruptData = useChatStore((state) => state.pendingInterruptData);
  const appendTransientMessage = useChatStore((state) => state.appendTransientMessage);
  const setTransientSessionId = useChatStore((state) => state.setTransientSessionId);
  const clearTransientMessages = useChatStore((state) => state.clearTransientMessages);
  const setInput = useChatStore((state) => state.setInput);
  const setIsLoading = useChatStore((state) => state.setIsLoading);
  const setCurrentSessionId = useChatStore((state) => state.setCurrentSessionId);
  const setCurrentThreadId = useChatStore((state) => state.setCurrentThreadId);
  const setCurrentExecutionTaskId = useChatStore((state) => state.setCurrentExecutionTaskId);
  const setSessionExecutionStatus = useChatStore((state) => state.setSessionExecutionStatus);
  const setPendingInterruptData = useChatStore((state) => state.setPendingInterruptData);
  const startStreaming = useChatStore((state) => state.startStreaming);
  const stopStreaming = useChatStore((state) => state.stopStreaming);
  const setExecutionStatus = useChatStore((state) => state.setExecutionStatus);
  const pushStreamingActivity = useChatStore((state) => state.pushStreamingActivity);
  const completeStreamingActivity = useChatStore((state) => state.completeStreamingActivity);
  const pushStreamingMessage = useChatStore((state) => state.pushStreamingMessage);

  const setProposalPreview = useUIStore((state) => state.setProposalPreview);

  const activeStreamRef = useRef<ActiveStreamState>({
    sessionId: null,
    threadId: null,
    lastEventId: null,
    executionStatus: null,
  });
  const abortControllerRef = useRef<AbortController | null>(null);
  const stoppedRef = useRef(false);

  const fetchProposal = useCallback(async (proposalId: number): Promise<WorkflowProposal | null> => {
    if (!workflowId) return null;
    try {
      return await backendApiClient.request<WorkflowProposal>(
        `/api/nexus/${workflowId}/proposals/${proposalId}`,
      );
    } catch (error) {
      console.warn('[useChatActions] Failed to fetch proposal:', error);
      return null;
    }
  }, [workflowId]);

  const handleAgentEnd = useCallback(async (data: Record<string, unknown>) => {
    const resolvedSessionId = activeStreamRef.current.sessionId ?? currentSessionId;
    const proposalId = data.proposal_id as number | undefined;
    const proposal = proposalId ? await fetchProposal(proposalId) : null;

    handleProposalResponse(proposal, undefined, setProposalPreview);

    const message = createAssistantMessage({
      response: (data.content as string) ?? '',
      proposal,
      proposalError: null,
      thinking: data.thinking as string[] | undefined,
      interruptRequired: false,
      interruptData: undefined,
      selectedModel,
    });

    setTransientSessionId(resolvedSessionId);
    appendTransientMessage(message);
    setIsLoading(false);
    setExecutionStatus('completed');
    setSessionExecutionStatus('completed');
    setPendingInterruptData(null);
    stopStreaming();
    queryClient.invalidateQueries({ queryKey: chatKeys.sessionsByWorkflow(workflowId) });
    if (resolvedSessionId !== null) {
      queryClient.invalidateQueries({ queryKey: chatKeys.messagesBySession(resolvedSessionId) });
    }
  }, [
    appendTransientMessage,
    currentSessionId,
    fetchProposal,
    queryClient,
    selectedModel,
    setExecutionStatus,
    setIsLoading,
    setPendingInterruptData,
    setProposalPreview,
    setSessionExecutionStatus,
    setTransientSessionId,
    stopStreaming,
    workflowId,
  ]);

  const handleInterrupt = useCallback((data: Record<string, unknown>, aiContent?: string) => {
    const resolvedSessionId = activeStreamRef.current.sessionId ?? currentSessionId;
    const interruptData: JsonObject = {
      type: data.type as string,
      clarification_questions: { questions: data.questions as JsonObject[] },
    };

    const message = createAssistantMessage({
      response: aiContent ?? '',
      proposal: null,
      proposalError: null,
      thinking: undefined,
      interruptRequired: true,
      interruptData,
      selectedModel,
    });

    setTransientSessionId(resolvedSessionId);
    appendTransientMessage(message);
    setIsLoading(false);
    setExecutionStatus('interrupted');
    setSessionExecutionStatus('interrupted');
    setPendingInterruptData(interruptData as ChatMessage['interruptData']);
    stopStreaming();
    queryClient.invalidateQueries({ queryKey: chatKeys.sessionsByWorkflow(workflowId) });
    if (resolvedSessionId !== null) {
      queryClient.invalidateQueries({ queryKey: chatKeys.messagesBySession(resolvedSessionId) });
    }
  }, [
    appendTransientMessage,
    currentSessionId,
    queryClient,
    selectedModel,
    setExecutionStatus,
    setIsLoading,
    setPendingInterruptData,
    setSessionExecutionStatus,
    setTransientSessionId,
    stopStreaming,
    workflowId,
  ]);

  // eslint-disable-next-line complexity
  const processSSEStream = useCallback(async (response: Response, isReconnecting = false) => {
    let lastAiMessageContent = '';
    stoppedRef.current = false;

    try {
      for await (const sseEvent of readSSEStream(response)) {
        if (stoppedRef.current) return;
        const { type, data, session_id } = sseEvent.data;

        if (sseEvent.id) {
          activeStreamRef.current.lastEventId = sseEvent.id;
        }

        if (type === 'session_info') {
          if (!isReconnecting) {
            const resolvedSessionId = session_id ?? (data.session_id as number | undefined);
            const resolvedThreadId = data.thread_id as string | undefined;
            const resolvedExecutionTaskId = data.execution_task_id as string | undefined;

            updateSessionIds({
              sessionId: resolvedSessionId,
              threadId: resolvedThreadId,
              currentSessionId,
              currentThreadId,
              setCurrentSessionId,
              setCurrentThreadId,
            });

            patchStreamRef(activeStreamRef, {
              sessionId: resolvedSessionId ?? null,
              threadId: resolvedThreadId ?? currentThreadId ?? '',
              executionStatus: 'queued',
            }, workflowId);

            setTransientSessionId(resolvedSessionId ?? null);
            setCurrentExecutionTaskId(resolvedExecutionTaskId ?? null);
            setSessionExecutionStatus('queued');
            setPendingInterruptData(null);
          }
        } else if (type === 'agent_start') {
          setExecutionStatus('running');
          setSessionExecutionStatus('running');
          if (!isReconnecting) {
            patchStreamRef(activeStreamRef, { executionStatus: 'running' }, workflowId);
          }
        } else if (type === 'tool_start') {
          pushStreamingActivity((data.tool_name as string) ?? 'unknown_tool');
        } else if (type === 'tool_end') {
          completeStreamingActivity(data.tool_name as string);
        } else if (type === 'ai_message') {
          const content = data.content as string;
          pushStreamingMessage(content);
          lastAiMessageContent = content;
        } else if (type === 'agent_end') {
          if (isReconnecting) {
            const proposal = (data.proposal_id as number | undefined)
              ? await fetchProposal(data.proposal_id as number)
              : null;
            handleProposalResponse(proposal, undefined, setProposalPreview);
            setIsLoading(false);
            setExecutionStatus('completed');
            setSessionExecutionStatus('completed');
            setPendingInterruptData(null);
            stopStreaming();
          } else {
            await handleAgentEnd(data);
          }
          finalizeStreamSession(activeStreamRef, workflowId);
        } else if (type === 'interrupt') {
          if (isReconnecting) {
            setIsLoading(false);
            setExecutionStatus('interrupted');
            setSessionExecutionStatus('interrupted');
            stopStreaming();
          } else {
            handleInterrupt(data, lastAiMessageContent);
          }
          finalizeStreamSession(activeStreamRef, workflowId);
        } else if (type === 'error') {
          if (!isReconnecting) {
            appendTransientMessage(
              createErrorMessage({ message: (data.message as string) ?? 'An error occurred' }),
            );
          }
          setProposalPreview(null);
          setIsLoading(false);
          setSessionExecutionStatus('failed');
          setPendingInterruptData(null);
          stopStreaming();
          finalizeStreamSession(activeStreamRef, workflowId);
        } else if (type === 'done') {
          stopStreaming();
        }
      }
    } catch (error) {
      if ((error as { name?: string })?.name === 'AbortError') return;
      console.error('[useChatActions] SSE stream error:', error);
      if (!isReconnecting) {
        appendTransientMessage(createErrorMessage(error));
      } else {
        queryClient.invalidateQueries({ queryKey: chatKeys.sessionsByWorkflow(workflowId) });
      }
      setProposalPreview(null);
    } finally {
      setIsLoading(false);
      stopStreaming();
    }
  }, [
    workflowId,
    currentSessionId,
    currentThreadId,
    setCurrentSessionId,
    setCurrentThreadId,
    setTransientSessionId,
    setCurrentExecutionTaskId,
    setExecutionStatus,
    setSessionExecutionStatus,
    setPendingInterruptData,
    pushStreamingActivity,
    completeStreamingActivity,
    pushStreamingMessage,
    appendTransientMessage,
    setIsLoading,
    setProposalPreview,
    stopStreaming,
    handleAgentEnd,
    handleInterrupt,
    fetchProposal,
    queryClient,
  ]);

  const reconnectToStream = useCallback(async (sessionId: number, lastEventId: string | null) => {
    if (!workflowId) return;
    if (activeStreamRef.current.sessionId !== null) return;

    activeStreamRef.current = { ...activeStreamRef.current, sessionId };

    setIsLoading(true);
    startStreaming(sessionId);
    try {
      const headers: HeadersInit = lastEventId ? { 'Last-Event-ID': lastEventId } : {};
      const response = await backendApiClient.stream(
        `/api/nexus/${workflowId}/chat/stream/${sessionId}`,
        { method: 'GET', headers },
      );
      await processSSEStream(response, true);
      await queryClient.refetchQueries({ queryKey: chatKeys.messagesBySession(sessionId) });
      clearTransientMessages();
    } catch (error) {
      console.error('[useChatActions] Stream reconnect failed:', error);
      setIsLoading(false);
      stopStreaming();
      clearPersistedSession(workflowId);
      activeStreamRef.current = { sessionId: null, threadId: null, lastEventId: null, executionStatus: null };
      queryClient.invalidateQueries({ queryKey: chatKeys.messagesBySession(sessionId) });
      queryClient.invalidateQueries({ queryKey: chatKeys.sessionsByWorkflow(workflowId) });
    }
  }, [
    clearTransientMessages,
    workflowId,
    setIsLoading,
    startStreaming,
    stopStreaming,
    processSSEStream,
    queryClient,
  ]);

  useEffect(() => {
    if (!workflowId || currentSessionId !== null) return;
    const saved = restoreChatSession(workflowId);
    if (!saved) return;

    setCurrentSessionId(saved.sessionId);
    setCurrentThreadId(saved.threadId);

    if (saved.executionStatus === 'running' || saved.executionStatus === 'queued') {
      reconnectToStream(saved.sessionId, saved.lastEventId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [workflowId]);

  const submitResume = useCallback(async ({
    userContent,
    answers,
    message,
  }: {
    userContent: string;
    answers?: ClarificationAnswers;
    message?: string;
  }) => {
    if (!workflowId || !currentThreadId || isLoading || !pendingInterruptData) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: userContent,
      timestamp: new Date(),
      model: selectedModel,
    };

    appendTransientMessage(userMessage);
    setTransientSessionId(currentSessionId);
    setIsLoading(true);
    startStreaming(0);

    try {
      const requestId = createRequestId();
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const response = await resumeWorkflowChat({
        workflowId,
        threadId: currentThreadId,
        answers,
        message,
        requestId,
      });

      await processSSEStream(response);
    } catch (error) {
      if ((error as { name?: string })?.name === 'AbortError') return;
      console.error('Failed to resume interrupted chat:', error);
      appendTransientMessage(createErrorMessage(error));
      setProposalPreview(null);
      setIsLoading(false);
      stopStreaming();
    }
  }, [
    workflowId,
    currentSessionId,
    currentThreadId,
    isLoading,
    pendingInterruptData,
    selectedModel,
    appendTransientMessage,
    setTransientSessionId,
    setIsLoading,
    setProposalPreview,
    startStreaming,
    stopStreaming,
    processSSEStream,
  ]);

  const handleResumeMessage = useCallback(async () => {
    if (!input.trim()) return;

    const messageContent = input.trim();
    setInput('');

    await submitResume({
      userContent: messageContent,
      message: messageContent,
    });
  }, [input, setInput, submitResume]);

  const handleSend = useCallback(async () => {
    if (!input.trim() || !workflowId || isLoading) return;

    const messageContent = input.trim();
    setInput('');

    const userMessage: ChatMessage = {
      role: 'user',
      content: messageContent,
      timestamp: new Date(),
      model: selectedModel,
    };

    appendTransientMessage(userMessage);
    setTransientSessionId(currentSessionId);
    setIsLoading(true);
    startStreaming(0);

    try {
      const requestId = createRequestId();
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      const response = await backendApiClient.stream(`/api/nexus/${workflowId}/chat`, {
        method: 'POST',
        signal: controller.signal,
        body: {
          message: messageContent,
          model: selectedModel,
          session_id: currentSessionId || undefined,
          thread_id: currentThreadId || undefined,
          request_id: requestId,
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
          workflowState: prepareWorkflowState(nodes, edges),
        },
      });
      await processSSEStream(response);
    } catch (error) {
      if ((error as { name?: string })?.name === 'AbortError') return;
      console.error('Failed to send chat message:', error);
      appendTransientMessage(createErrorMessage(error));
      setProposalPreview(null);
      setIsLoading(false);
      stopStreaming();
    }
  }, [
    input,
    workflowId,
    isLoading,
    selectedModel,
    currentSessionId,
    currentThreadId,
    nodes,
    edges,
    appendTransientMessage,
    setInput,
    setIsLoading,
    setProposalPreview,
    setTransientSessionId,
    startStreaming,
    stopStreaming,
    processSSEStream,
  ]);

  const handleAnswerClarifications = useCallback(async (answers: ClarificationAnswers) => {
    if (!workflowId || !currentThreadId || isLoading || !pendingInterruptData) return;

    const answerLines = answers.answers.map((answer) => {
      const summary = `Q(${answer.question_id}): ${answer.selected_values.join(', ')}`;
      return answer.custom_input ? `${summary} (Custom: ${answer.custom_input})` : summary;
    });
    const userContent = `Answered ${answers.answers.length} questions:\n${answerLines.join('\n')}`;

    await submitResume({
      userContent,
      answers,
    });
  }, [workflowId, currentThreadId, isLoading, pendingInterruptData, submitResume]);

  const handleAnswerClarification = useCallback(async (answer: ClarificationAnswer) => {
    await handleAnswerClarifications({ answers: [answer] });
  }, [handleAnswerClarifications]);

  const handleNewSession = useCallback(async () => {
    if (!workflowId) return;
    try {
      const response = await backendApiClient.request<ChatSession>(
        `/api/nexus/${workflowId}/chat/sessions`,
        {
          method: 'POST',
          body: JSON.stringify({ title: null }),
        },
      );
      setCurrentSessionId(response.id);
      setCurrentThreadId(response.thread_id);
      setCurrentExecutionTaskId(response.current_execution_task_id ?? null);
      setSessionExecutionStatus(response.current_execution_status ?? null);
      setPendingInterruptData(response.pending_interrupt_data ?? null);
      clearTransientMessages();
      setProposalPreview(null);
      setIsLoading(false);
      stopStreaming();
      clearPersistedSession(workflowId);
      queryClient.invalidateQueries({ queryKey: chatKeys.sessionsByWorkflow(workflowId) });
    } catch (error) {
      console.error('Failed to create session:', error);
    }
  }, [
    workflowId,
    setCurrentSessionId,
    setCurrentThreadId,
    setCurrentExecutionTaskId,
    setSessionExecutionStatus,
    setPendingInterruptData,
    clearTransientMessages,
    setProposalPreview,
    setIsLoading,
    stopStreaming,
    queryClient,
  ]);

  const handleSelectSession = useCallback((sessionId: number, sessions: ChatSession[]) => {
    setCurrentSessionId(sessionId);

    const session = sessions.find((item) => item.id === sessionId);
    setCurrentThreadId(session?.thread_id ?? null);
    setCurrentExecutionTaskId(session?.current_execution_task_id ?? null);
    setSessionExecutionStatus(session?.current_execution_status ?? null);
    setPendingInterruptData(session?.pending_interrupt_data ?? null);
    clearTransientMessages();
    setProposalPreview(null);
    stopStreaming();
    setIsLoading(false);

    if (workflowId) {
      persistSelectedSession(workflowId, sessionId, session?.thread_id ?? null);
    }
  }, [
    workflowId,
    setCurrentSessionId,
    setCurrentThreadId,
    setCurrentExecutionTaskId,
    setSessionExecutionStatus,
    setPendingInterruptData,
    clearTransientMessages,
    setProposalPreview,
    stopStreaming,
    setIsLoading,
  ]);

  const handleStop = useCallback(() => {
    stoppedRef.current = true;
    stopStreaming();
    setIsLoading(false);
    appendTransientMessage({
      role: 'assistant',
      content: 'Stopped. How can I help?',
      timestamp: new Date(),
    });
  }, [stopStreaming, setIsLoading, appendTransientMessage]);

  return {
    handleSend,
    handleStop,
    handleResumeMessage,
    handleAnswerClarification,
    handleAnswerClarifications,
    handleNewSession,
    handleSelectSession,
  };
}
