import { useEffect } from 'react';

import { useChatSessions } from './useChatSessions';
import { useChatMessages } from './useChatMessages';
import { useChatStore } from '@/stores';
import type { ChatMessage } from '@/components/workflows/buildtypes';

function findLatestPendingInterrupt(messages: ChatMessage[]): ChatMessage['interruptData'] | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === 'assistant' && msg.interruptRequired && msg.interruptData) {
      return msg.interruptData;
    }
  }
  return null;
}

export function useChatSessionData(workflowId: string | null) {
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const setCurrentExecutionTaskId = useChatStore((state) => state.setCurrentExecutionTaskId);
  const setSessionExecutionStatus = useChatStore((state) => state.setSessionExecutionStatus);
  const setPendingInterruptData = useChatStore((state) => state.setPendingInterruptData);
  const sessionsQuery = useChatSessions(workflowId);
  const sessions = sessionsQuery.data?.pages.flatMap((page) => page) ?? [];
  const { data: sessionData } = useChatMessages(workflowId, currentSessionId);

  useEffect(() => {
    if (!sessionData) return;

    const restoredPendingInterrupt = sessionData.pendingInterruptData ?? findLatestPendingInterrupt(sessionData.messages);

    setCurrentExecutionTaskId(sessionData.currentExecutionTaskId);
    setSessionExecutionStatus(sessionData.currentExecutionStatus);
    setPendingInterruptData(restoredPendingInterrupt);
  }, [
    sessionData,
    setCurrentExecutionTaskId,
    setSessionExecutionStatus,
    setPendingInterruptData,
  ]);

  return { sessions, sessionsQuery };
}
