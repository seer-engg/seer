import { useEffect, useRef, useState } from 'react';

import { ScrollArea } from '@/components/ui/scroll-area';

import type { ChatSession, ModelInfo } from '../buildtypes';
import type { ClarificationAnswer, ClarificationAnswers } from '@/types/discovery';
import { ChatHeader } from './ChatHeader';
import { ChatInput } from './ChatInput';
import { MessagesList } from './MessagesList';
import type { SessionsStatus } from './types';
import { useChatStore } from '@/stores';
import { useWorkflowChatMessages } from '@/hooks/useWorkflowChatMessages';

interface ChatPanelProps {
  workflowId: string | null;
  onSend: () => void;
  onResumeSend?: () => void;
  onAnswerClarification?: (answer: ClarificationAnswer) => void;
  onAnswerClarifications?: (answers: ClarificationAnswers) => void;
  models: ModelInfo[];
  isLoadingModels: boolean;
  filterSystemPrompt: (content: string) => string;
  onAcceptProposal: (proposalId: number) => void;
  onRejectProposal: (proposalId: number) => void;
  activePreviewProposalId?: number | null;
  // Optional — when provided, renders the session management header
  onStop?: () => void;
  onNewSession?: () => void;
  sessions?: ChatSession[];
  sessionsStatus?: SessionsStatus;
  onSelectSession?: (sessionId: number) => void;
}

export function ChatPanel({
  workflowId,
  onSend,
  onResumeSend,
  onAnswerClarification,
  onAnswerClarifications,
  models,
  isLoadingModels,
  filterSystemPrompt,
  onAcceptProposal,
  onRejectProposal,
  activePreviewProposalId,
  onStop,
  onNewSession,
  sessions,
  sessionsStatus,
  onSelectSession,
}: ChatPanelProps) {
  const [sessionPopoverOpen, setSessionPopoverOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages } = useWorkflowChatMessages(workflowId);
  const messageCount = messages.length;
  const isLoading = useChatStore((state) => state.isLoading);
  const pendingInterruptData = useChatStore((state) => state.pendingInterruptData);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messageCount]);

  const latestMessageInterruptData = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant' && message.interruptRequired && message.interruptData)
    ?.interruptData;
  const hasPendingInterrupt = Boolean(pendingInterruptData || latestMessageInterruptData);
  const showSessionHeader = Boolean(onNewSession && sessions && sessionsStatus && onSelectSession);
  const inputSubmitHandler = hasPendingInterrupt && onResumeSend ? onResumeSend : onSend;

  if (!workflowId) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p className="text-sm">Save a workflow to start chatting</p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-w-0 flex-col bg-background">
      {showSessionHeader && (
        <ChatHeader
          onNewSession={onNewSession!}
          sessionPopoverOpen={sessionPopoverOpen}
          onSessionPopoverOpenChange={setSessionPopoverOpen}
          sessions={sessions!}
          sessionsStatus={sessionsStatus!}
          onSelectSession={(sessionId) => {
            onSelectSession!(sessionId);
            setSessionPopoverOpen(false);
          }}
        />
      )}
      <ScrollArea className="min-w-0 flex-1 px-3 py-3">
        <MessagesList
          messages={messages}
          filterSystemPrompt={filterSystemPrompt}
          listEndRef={messagesEndRef}
          onAcceptProposal={onAcceptProposal}
          onRejectProposal={onRejectProposal}
          activePreviewProposalId={activePreviewProposalId}
          onStartFreshChat={onNewSession}
          onAnswerClarification={onAnswerClarification}
          onAnswerClarifications={onAnswerClarifications}
          isClarificationLoading={isLoading}
        />
      </ScrollArea>
      <ChatInput
        onSend={inputSubmitHandler}
        onStop={onStop}
        models={models}
        isLoadingModels={isLoadingModels}
        isInterruptPending={hasPendingInterrupt}
      />
    </div>
  );
}
