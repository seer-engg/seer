import { useEffect, useMemo } from 'react';

import type { ChatMessage, WorkflowProposal } from '@/components/workflows/buildtypes';
import { workflowSpecToGraph } from '@/lib/workflow-graph';
import { useChatStore, useUIStore } from '@/stores';

import { useChatMessages } from './useChatMessages';

const EMPTY_MESSAGES: ChatMessage[] = [];

function findPendingProposal(messages: ChatMessage[]): WorkflowProposal | null {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.proposal?.status === 'pending') {
      return message.proposal;
    }
  }
  return null;
}

function areTimestampsClose(a: ChatMessage, b: ChatMessage) {
  return Math.abs(a.timestamp.getTime() - b.timestamp.getTime()) < 10_000;
}

// For interrupt messages, match on interrupt type rather than content
// because the transient and persisted versions may format content differently.
function interruptMessagesMatch(persisted: ChatMessage, transient: ChatMessage) {
  const persistedInterrupt = persisted.interruptRequired ?? false;
  const transientInterrupt = transient.interruptRequired ?? false;
  if (!persistedInterrupt || !transientInterrupt) {
    return null;
  }
  const persistedType = persisted.interruptData?.type;
  const transientType = transient.interruptData?.type;
  if (persistedType && persistedType === transientType) {
    return areTimestampsClose(persisted, transient);
  }
  return null;
}

function messageMatchesPersisted(persisted: ChatMessage, transient: ChatMessage) {
  if (persisted.role !== transient.role) {
    return false;
  }

  const interruptResult = interruptMessagesMatch(persisted, transient);
  if (interruptResult !== null) {
    return interruptResult;
  }

  if (persisted.content !== transient.content) {
    return false;
  }
  const persistedInterrupt = persisted.interruptRequired ?? false;
  const transientInterrupt = transient.interruptRequired ?? false;
  if (persistedInterrupt !== transientInterrupt) {
    return false;
  }
  if ((persisted.proposal?.id ?? null) !== (transient.proposal?.id ?? null)) {
    return false;
  }

  return areTimestampsClose(persisted, transient);
}

function getMatchedTransientIndexes(
  persistedMessages: ChatMessage[],
  transientMessages: ChatMessage[],
) {
  const matchedTransientIndexes = new Set<number>();
  let searchStartIndex = 0;

  transientMessages.forEach((transient, transientIndex) => {
    for (let persistedIndex = searchStartIndex; persistedIndex < persistedMessages.length; persistedIndex += 1) {
      if (!messageMatchesPersisted(persistedMessages[persistedIndex], transient)) {
        continue;
      }

      matchedTransientIndexes.add(transientIndex);
      searchStartIndex = persistedIndex + 1;
      break;
    }
  });

  return matchedTransientIndexes;
}

export function useWorkflowChatMessages(workflowId: string | null) {
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const transientMessages = useChatStore((state) => state.transientMessages);
  const transientSessionId = useChatStore((state) => state.transientSessionId);
  const clearTransientMessages = useChatStore((state) => state.clearTransientMessages);
  const setProposalPreview = useUIStore((state) => state.setProposalPreview);
  const messagesQuery = useChatMessages(workflowId, currentSessionId);

  const persistedMessages = useMemo(() => messagesQuery.data?.messages ?? EMPTY_MESSAGES, [messagesQuery.data]);
  const activeTransientMessages = useMemo(() => {
    if (transientMessages.length === 0) {
      return [];
    }
    if (currentSessionId === null) {
      return transientMessages;
    }
    return transientSessionId === currentSessionId ? transientMessages : [];
  }, [currentSessionId, transientMessages, transientSessionId]);

  const messages = useMemo(() => {
    if (activeTransientMessages.length === 0) {
      return persistedMessages;
    }

    const matchedTransientIndexes = getMatchedTransientIndexes(persistedMessages, activeTransientMessages);
    const remainingTransient = activeTransientMessages.filter(
      (_, transientIndex) => !matchedTransientIndexes.has(transientIndex),
    );

    return [...persistedMessages, ...remainingTransient];
  }, [activeTransientMessages, persistedMessages]);

  useEffect(() => {
    if (activeTransientMessages.length === 0) {
      return;
    }

    const matchedTransientIndexes = getMatchedTransientIndexes(persistedMessages, activeTransientMessages);
    const allPersisted = matchedTransientIndexes.size === activeTransientMessages.length;

    if (allPersisted) {
      clearTransientMessages();
    }
  }, [activeTransientMessages, clearTransientMessages, persistedMessages]);

  useEffect(() => {
    const pendingProposal = findPendingProposal(messages);

    if (pendingProposal?.spec) {
      try {
        setProposalPreview({
          proposal: pendingProposal,
          graph: workflowSpecToGraph(pendingProposal.spec),
        });
        return;
      } catch (error) {
        console.error('Failed to restore workflow preview from proposal:', error);
      }
    }

    setProposalPreview(null);
  }, [messages, setProposalPreview]);

  return {
    messages,
    messagesQuery,
  };
}
