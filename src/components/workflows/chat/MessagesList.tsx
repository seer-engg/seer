import { useMemo, useState } from 'react';
import type { RefObject } from 'react';
import { Bot, Check, Loader2 } from 'lucide-react';

import { MessageBubble } from './MessageBubble';
import { useChatStore } from '@/stores';
import type { ChatMessage, StreamActivity } from '@/components/workflows/buildtypes';
import type { ClarificationAnswer, ClarificationAnswers } from '@/types/discovery';

function StreamingBubble({ activities }: { activities: StreamActivity[] }) {
  return (
    <div className="flex w-full min-w-0 gap-3 justify-start">
      <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center text-muted-foreground mt-1">
        <Bot className="w-4 h-4" />
      </div>
      <div className="min-w-0 w-fit max-w-[calc(100%-0.75rem)] flex-1 sm:max-w-[92%]">
        <div className="bg-muted rounded-lg p-3">
          {activities.length > 0 ? (
            <div className="space-y-1">
              {activities.map((activity, i) => {
                if (activity.type === 'ai_message') {
                  return (
                    <p key={i} className="text-xs text-muted-foreground italic opacity-75">
                      {activity.content}
                    </p>
                  );
                }
                return (
                  <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
                    {activity.status === 'done' ? (
                      <Check className="w-3 h-3 text-emerald-500 flex-shrink-0" />
                    ) : (
                      <Loader2 className="w-3 h-3 animate-spin flex-shrink-0" />
                    )}
                    <span className={activity.status === 'done' ? 'opacity-60' : ''}>
                      {activity.tool}
                    </span>
                  </div>
                );
              })}
              {!activities.some((a) => a.type === 'tool' && a.status === 'running') && (
                <div className="flex gap-1 pt-1">
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-75" />
                  <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-150" />
                </div>
              )}
            </div>
          ) : (
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" />
              <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-75" />
              <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce delay-150" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface MessagesListProps {
  messages: ChatMessage[];
  filterSystemPrompt: (content: string) => string;
  listEndRef: RefObject<HTMLDivElement>;
  onAcceptProposal: (proposalId: number) => void;
  onRejectProposal: (proposalId: number) => void;
  activePreviewProposalId?: number | null;
  onStartFreshChat?: () => void;
  onAnswerClarification?: (answer: ClarificationAnswer) => void;
  onAnswerClarifications?: (answers: ClarificationAnswers) => void;
  isClarificationLoading?: boolean;
}

export function MessagesList({
  messages,
  filterSystemPrompt,
  listEndRef,
  onAcceptProposal,
  onRejectProposal,
  activePreviewProposalId,
  onStartFreshChat,
  onAnswerClarification,
  onAnswerClarifications,
  isClarificationLoading = false,
}: MessagesListProps) {
  const isLoading = useChatStore((state) => state.isLoading);
  const streamingActivities = useChatStore((state) => state.streamingActivities);
  const proposalActionLoading = useChatStore((state) => state.proposalActionLoading);
  const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());

  const noMessages = useMemo(() => messages.length === 0, [messages.length]);

  const toggleThinking = (index: number) => {
    setExpandedThinking((prev) => {
      const copy = new Set(prev);
      if (copy.has(index)) {
        copy.delete(index);
      } else {
        copy.add(index);
      }
      return copy;
    });
  };

  return (
    <div className="min-w-0 w-full space-y-4">
      {noMessages ? (
        <div className="text-center text-sm text-muted-foreground py-8">
        </div>
      ) : (
        messages.map((message, index) => {
          const filteredContent = filterSystemPrompt(message.content);
          return (
            <MessageBubble
              key={index}
              message={message}
              filteredContent={filteredContent}
              isThinkingExpanded={expandedThinking.has(index)}
              onToggleThinking={() => toggleThinking(index)}
              onAcceptProposal={onAcceptProposal}
              onRejectProposal={onRejectProposal}
              proposalActionLoading={proposalActionLoading}
              isActivePreview={Boolean(
                activePreviewProposalId && message.proposal?.id === activePreviewProposalId,
              )}
              onStartFreshChat={onStartFreshChat}
              onAnswerClarification={onAnswerClarification}
              onAnswerClarifications={onAnswerClarifications}
              isClarificationLoading={isClarificationLoading}
            />
          );
        })
      )}
      {isLoading && <StreamingBubble activities={streamingActivities} />}
      <div ref={listEndRef} />
    </div>
  );
}
