import { useCallback, useMemo, useState } from 'react';
import { AlertCircle, Check, ChevronDown, ChevronUp, Copy, X } from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

import { cn } from '@/lib/utils';

import type { ChatMessage, WorkflowProposal } from '../buildtypes';
import type { ClarificationAnswer, ClarificationAnswers, ClarificationQuestion } from '@/types/discovery';
import { ChatCostCapError } from './ChatCostCapError';
import { ClarificationQuestionPanel } from '../discovery/ClarificationQuestionPanel';
import { ClarificationQuestionsPanel } from '../discovery/ClarificationQuestionsPanel';
import { MarkdownContent } from './MarkdownContent';

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [content]);

  return (
    <button onClick={handleCopy} className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:bg-muted">
      {copied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5 text-muted-foreground" />}
    </button>
  );
}

interface MessageBubbleProps {
  message: ChatMessage;
  filteredContent: string;
  isThinkingExpanded: boolean;
  onToggleThinking: () => void;
  onAcceptProposal: (proposalId: number) => void;
  onRejectProposal: (proposalId: number) => void;
  proposalActionLoading: number | null;
  isActivePreview?: boolean;
  onStartFreshChat?: () => void;
  /** Legacy: single question callback (kept for backward compatibility) */
  onAnswerClarification?: (answer: ClarificationAnswer) => void;
  /** New: multiple questions callback */
  onAnswerClarifications?: (answers: ClarificationAnswers) => void;
  isClarificationLoading?: boolean;
}

interface ThinkingSectionProps {
  thinking: string[];
  isExpanded: boolean;
  onToggle: () => void;
}

function ThinkingSection({ thinking, isExpanded, onToggle }: ThinkingSectionProps) {
  return (
    <div className="mt-2 pt-2 border-t border-border/50">
      <button
        onClick={onToggle}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        <span>Thinking ({thinking.length} steps)</span>
      </button>
      {isExpanded && (
        <div className="mt-2 space-y-1 text-xs text-muted-foreground bg-background/50 p-2 rounded text-left">
          {thinking.map((step, idx) => (
            <div key={idx} className="font-mono">
              {step}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface ProposalSectionProps {
  proposal: WorkflowProposal;
  isActivePreview: boolean;
  proposalActionLoading: number | null;
  hasError: boolean;
  onAccept: (id: number) => void;
  onReject: (id: number) => void;
}

function ProposalSection({
  proposal,
  isActivePreview,
  proposalActionLoading,
  hasError,
  onAccept,
  onReject,
}: ProposalSectionProps) {
  const isLoading = proposalActionLoading === proposal.id;
  const isDisabled = isLoading || hasError;

  return (
    <div className="mt-3 pt-3 border-t border-border/50 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium">AI Proposal</p>
          <p className="text-xs text-muted-foreground">{proposal.summary}</p>
        </div>
        <Badge
          className={cn(
            'capitalize',
            proposal.status === 'pending' && 'bg-amber-100 text-amber-900',
            proposal.status === 'accepted' && 'bg-emerald-100 text-emerald-900',
            proposal.status === 'rejected' && 'bg-rose-100 text-rose-900',
          )}
        >
          {proposal.status}
        </Badge>
      </div>
      {isActivePreview && (
        <Alert className="bg-sky-100 text-sky-900 border-sky-200">
          <AlertTitle className="text-xs font-semibold">Preview Active</AlertTitle>
          <AlertDescription className="text-xs">
            This proposal is currently rendered on the canvas. Accept or reject to continue editing.
          </AlertDescription>
        </Alert>
      )}
      {proposal.status === 'pending' && (
        <div className="flex flex-col gap-2 sm:flex-row">
          <Button size="sm" className="flex-1" disabled={isDisabled} onClick={() => onAccept(proposal.id)}>
            <Check className="w-3 h-3 mr-2" />
            Accept
          </Button>
          <Button size="sm" variant="outline" className="flex-1" disabled={isDisabled} onClick={() => onReject(proposal.id)}>
            <X className="w-3 h-3 mr-2" />
            Reject
          </Button>
        </div>
      )}
    </div>
  );
}

interface CostCapErrorSectionProps {
  error: ChatMessage['error'];
  onStartFreshChat?: () => void;
}

function CostCapErrorSection({ error, onStartFreshChat }: CostCapErrorSectionProps) {
  if (error?.type !== 'cost_cap_exceeded' || !error.details) {
    return null;
  }

  const handleIncreaseLimitClick = () => {
    console.log('Navigate to settings to increase cost limit');
  };

  const handleStartFreshClick = () => {
    onStartFreshChat?.();
  };

  return (
    <div className="flex justify-start">
      <div className="min-w-0 w-fit max-w-[calc(100%-0.75rem)] sm:max-w-[92%]">
        <ChatCostCapError
          accumulatedCost={error.details.accumulated_cost ?? 0}
          costCap={error.details.cost_cap ?? 0}
          onIncreaseLimitClick={handleIncreaseLimitClick}
          onStartFreshClick={handleStartFreshClick}
        />
      </div>
    </div>
  );
}

interface MessageContentBubbleProps {
  content: string | null;
  thinking?: string[];
  proposalError?: string;
  proposal?: WorkflowProposal;
  isThinkingExpanded: boolean;
  onToggleThinking: () => void;
  isActivePreview: boolean;
  proposalActionLoading: number | null;
  onAcceptProposal: (id: number) => void;
  onRejectProposal: (id: number) => void;
  role: 'user' | 'assistant';
}

function MessageContentBubble({
  content,
  thinking,
  proposalError,
  proposal,
  isThinkingExpanded,
  onToggleThinking,
  isActivePreview,
  proposalActionLoading,
  onAcceptProposal,
  onRejectProposal,
  role,
}: MessageContentBubbleProps) {
  return (
    <div className={cn(
      'group relative w-fit min-w-0 max-w-full overflow-hidden rounded-2xl px-4 py-2.5',
      role === 'user' ? 'bg-muted rounded-br-md' : 'bg-muted/50 rounded-bl-md'
    )}>
      {role === 'assistant' && (
        <CopyButton content={content || ''} />
      )}
      {content && <MarkdownContent content={content} />}

      {thinking && thinking.length > 0 && (
        <ThinkingSection thinking={thinking} isExpanded={isThinkingExpanded} onToggle={onToggleThinking} />
      )}

      {proposalError && (
        <div className="mt-3 pt-3 border-t border-border/50">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Proposal Validation Error</AlertTitle>
            <AlertDescription className="text-xs">{proposalError}</AlertDescription>
          </Alert>
        </div>
      )}

      {proposal && (
        <ProposalSection
          proposal={proposal}
          isActivePreview={isActivePreview}
          proposalActionLoading={proposalActionLoading}
          hasError={!!proposalError}
          onAccept={onAcceptProposal}
          onReject={onRejectProposal}
        />
      )}
    </div>
  );
}

function useMessageContent(message: ChatMessage, filteredContent: string) {
  return useMemo(() => {
    return message.role === 'assistant' ? (filteredContent || message.content) : message.content;
  }, [filteredContent, message.content, message.role]);
}

interface ClarificationState {
  hasQuestions: boolean;      // New plural format
  hasQuestion: boolean;       // Legacy singular format
  hasClarification: boolean;  // Either format
}

/** Extract questions array from interrupt data, supporting both nested and flat shapes. */
function getQuestionsFromInterrupt(interruptData: ChatMessage['interruptData']): ClarificationQuestion[] | undefined {
  if (interruptData?.type !== 'clarification_questions') return undefined;
  return interruptData.clarification_questions?.questions ?? (interruptData.questions as ClarificationQuestion[] | undefined);
}

function useClarificationState(message: ChatMessage): ClarificationState {
  return useMemo(() => {
    const questionsArray = getQuestionsFromInterrupt(message.interruptData);
    const hasQuestions = Boolean(message.interruptRequired && questionsArray?.length);

    const hasQuestion = Boolean(
      !hasQuestions &&
      message.interruptRequired &&
      message.interruptData?.type === 'clarification_question' &&
      message.interruptData.clarification_question
    );

    return {
      hasQuestions,
      hasQuestion,
      hasClarification: hasQuestions || hasQuestion,
    };
  }, [message.interruptRequired, message.interruptData]);
}

function shouldRenderMessage(message: ChatMessage, showContent: string, hasClarification: boolean): boolean {
  if (message.error?.type === 'cost_cap_exceeded') return true;
  if (message.role !== 'assistant') return true;
  return Boolean(showContent || hasClarification);
}

function hasMessageContent(
  showContent: string,
  thinking?: string[],
  proposal?: WorkflowProposal,
  proposalError?: string
): boolean {
  return Boolean(showContent || thinking?.length || proposal || proposalError);
}

interface ClarificationSectionProps {
  message: ChatMessage;
  clarificationState: ClarificationState;
  hasContentBubble: boolean;
  onAnswerClarification?: (answer: ClarificationAnswer) => void;
  onAnswerClarifications?: (answers: ClarificationAnswers) => void;
  isLoading: boolean;
}

function ClarificationSection({
  message,
  clarificationState,
  hasContentBubble,
  onAnswerClarification,
  onAnswerClarifications,
  isLoading,
}: ClarificationSectionProps) {
  const wrapperClass = hasContentBubble ? 'mt-3' : '';

  // New plural format: multiple clarification questions
  if (clarificationState.hasQuestions && onAnswerClarifications) {
    return (
      <div className={wrapperClass}>
        <ClarificationQuestionsPanel
          questions={getQuestionsFromInterrupt(message.interruptData)!}
          onSubmit={onAnswerClarifications}
          isLoading={isLoading}
        />
      </div>
    );
  }

  // Legacy singular format: single clarification question
  if (clarificationState.hasQuestion && onAnswerClarification) {
    return (
      <div className={wrapperClass}>
        <ClarificationQuestionPanel
          question={message.interruptData!.clarification_question!}
          onAnswer={onAnswerClarification}
          isLoading={isLoading}
        />
      </div>
    );
  }

  return null;
}

export function MessageBubble({
  message,
  filteredContent,
  isThinkingExpanded,
  onToggleThinking,
  onAcceptProposal,
  onRejectProposal,
  proposalActionLoading,
  isActivePreview,
  onStartFreshChat,
  onAnswerClarification,
  onAnswerClarifications,
  isClarificationLoading = false,
}: MessageBubbleProps) {
  const showContent = useMessageContent(message, filteredContent);
  const clarificationState = useClarificationState(message);

  if (message.error?.type === 'cost_cap_exceeded') {
    return <CostCapErrorSection error={message.error} onStartFreshChat={onStartFreshChat} />;
  }

  if (!shouldRenderMessage(message, showContent, clarificationState.hasClarification)) {
    return null;
  }

  const hasContentBubble = hasMessageContent(showContent, message.thinking, message.proposal, message.proposalError);

  const isUser = message.role === 'user';

  return (
    <div className={cn('flex w-full min-w-0', isUser ? 'justify-end' : 'justify-start')}>
      <div className="min-w-0 w-fit max-w-[calc(100%-0.75rem)] sm:max-w-[92%]">
        {hasContentBubble && (
          <MessageContentBubble
            content={showContent}
            thinking={message.thinking}
            proposalError={message.proposalError}
            proposal={message.proposal}
            isThinkingExpanded={isThinkingExpanded}
            onToggleThinking={onToggleThinking}
            isActivePreview={!!isActivePreview}
            proposalActionLoading={proposalActionLoading}
            onAcceptProposal={onAcceptProposal}
            onRejectProposal={onRejectProposal}
            role={message.role}
          />
        )}

        <ClarificationSection
          message={message}
          clarificationState={clarificationState}
          hasContentBubble={hasContentBubble}
          onAnswerClarification={onAnswerClarification}
          onAnswerClarifications={onAnswerClarifications}
          isLoading={isClarificationLoading}
        />
      </div>
    </div>
  );
}
