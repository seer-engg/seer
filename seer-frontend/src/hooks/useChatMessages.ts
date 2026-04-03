import { useQuery } from '@tanstack/react-query';

import { backendApiClient } from '@/lib/api-client';
import { chatKeys } from '@/lib/query-keys';
import { getDisplayableAssistantMessage } from '../components/workflows/utils';

import type { ChatMessage, WorkflowProposal } from '../components/workflows/buildtypes';

/**
 * Backend response type for session messages
 *
 * IMPORTANT: Must include interrupt_required and interrupt_data fields to preserve
 * clarification questions when refetching messages from the backend.
 *
 * Without these fields, clarification questions would be lost when switching sessions
 * or when useChatSessionData refetches messages.
 */
type SessionMessageResponse = {
  id: number;
  role: string;
  content: string;
  thinking?: string;
  proposal?: WorkflowProposal | null;
  created_at: string;
  // Interrupt/clarification fields - critical for preserving agent questions
  interrupt_required?: boolean;
  interrupt_data?: {
    type?: string;
    // Backend persists the flat format: { type, questions: [...] }
    questions?: import('@/types/discovery').ClarificationQuestion[];
    // Already-normalized format (e.g. written by a future backend version)
    clarification_questions?: { questions: import('@/types/discovery').ClarificationQuestion[] };
    clarification_question?: import('@/types/discovery').ClarificationQuestion;
    [key: string]: unknown;
  };
};

/**
 * Normalize interrupt_data from the backend's flat format to the nested format
 * that MessageBubble expects.
 *
 * Backend persists:   { type: 'clarification_questions', questions: [...] }
 * MessageBubble needs: { type: 'clarification_questions', clarification_questions: { questions: [...] } }
 *
 * handleInterrupt() performs this wrapping for live SSE events. This function
 * does the same for messages loaded from the DB.
 */
function normalizeInterruptData(
  raw: SessionMessageResponse['interrupt_data'],
): SessionMessageResponse['interrupt_data'] {
  if (!raw) return raw;
  if (
    (raw.type === 'clarification_questions' || raw.type === 'clarification_question') &&
    Array.isArray(raw.questions) &&
    !raw.clarification_questions
  ) {
    return { ...raw, clarification_questions: { questions: raw.questions } };
  }
  return raw;
}

type ChatSessionMessagesResponse = {
  id: number;
  current_execution_status?: import('@/components/workflows/buildtypes').ChatExecutionStatus;
  current_execution_task_id?: string;
  pending_interrupt_type?: string | null;
  pending_interrupt_data?: SessionMessageResponse['interrupt_data'] | null;
  messages: SessionMessageResponse[];
};

export type ChatSessionMessagesResult = {
  currentExecutionStatus: import('@/components/workflows/buildtypes').ChatExecutionStatus | null;
  currentExecutionTaskId: string | null;
  pendingInterruptType: string | null;
  pendingInterruptData: SessionMessageResponse['interrupt_data'] | null;
  messages: ChatMessage[];
};

/**
 * Fetches chat messages for a specific session from the backend
 *
 * This hook is used by useChatSessionData to keep messages in sync with the backend.
 * It's critical that this hook preserves ALL message fields, including interrupt_required
 * and interrupt_data, to prevent clarification questions from being lost during refetches.
 *
 * @param workflowId - The workflow ID
 * @param currentSessionId - The current chat session ID
 * @returns Query result with ChatMessage array
 */
export function useChatMessages(workflowId: string | null, currentSessionId: number | null) {
  return useQuery<ChatSessionMessagesResult>({
    queryKey: chatKeys.messagesBySession(currentSessionId),
    queryFn: async () => {
      if (!workflowId || !currentSessionId) {
        return {
          currentExecutionStatus: null,
          currentExecutionTaskId: null,
          pendingInterruptType: null,
          pendingInterruptData: null,
          messages: [],
        };
      }
      const response = await backendApiClient.request<ChatSessionMessagesResponse>(
        `/api/nexus/${workflowId}/chat/sessions/${currentSessionId}`,
        {
          method: 'GET',
        },
      );

      return {
        currentExecutionStatus: response.current_execution_status ?? null,
        currentExecutionTaskId: response.current_execution_task_id ?? null,
        pendingInterruptType: response.pending_interrupt_type ?? null,
        pendingInterruptData: normalizeInterruptData(response.pending_interrupt_data ?? undefined) ?? null,
        // Map backend messages to ChatMessage format
        // CRITICAL: Must preserve interrupt_required and interrupt_data fields
        // to prevent clarification questions from being lost
        messages: response.messages.map((msg) => ({
          role: msg.role as 'user' | 'assistant',
          content:
            msg.role === 'assistant'
              ? getDisplayableAssistantMessage(
                  msg.content,
                  msg.proposal?.summary,
                  msg.thinking ? msg.thinking.split('\n') : undefined,
                )
              : msg.content,
          thinking: msg.thinking ? msg.thinking.split('\n') : undefined,
          proposal: msg.proposal || undefined,
          proposalError: undefined,
          timestamp: new Date(msg.created_at),
          // Preserve interrupt data for clarification questions.
          // normalizeInterruptData converts the backend's flat { type, questions } format
          // to the nested { type, clarification_questions: { questions } } that MessageBubble expects.
          interruptRequired: msg.interrupt_required,
          interruptData: normalizeInterruptData(msg.interrupt_data),
        })),
      };
    },
    enabled: !!currentSessionId && !!workflowId,
  });
}
