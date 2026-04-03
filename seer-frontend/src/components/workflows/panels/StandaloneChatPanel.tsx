import { useState, useRef, useEffect, useCallback } from 'react';
import { FileEdit, Trash2, Check, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ChatPanel } from '../chat/ChatPanel';
import { DeleteWorkflowDialog } from '../discovery/DeleteWorkflowDialog';
import { SessionPopover } from '../chat/SessionPopover';
import { useChatActions } from '../../../hooks/useChatActions';
import { useProposalActions } from '../../../hooks/useProposalActions';
import { useAvailableModels } from '../../../hooks/useAvailableModels';
import { useChatSessionData } from '../../../hooks/useChatSessionData';
import { useInitialChatMessage } from '../../../hooks/useInitialChatMessage';
import { filterSystemPrompt } from '../utils';
import { useCanvasStore, useUIStore } from '@/stores';
import { useChatStore } from '@/stores/chatStore';
import type { ChatSession } from '../buildtypes';
import type { SessionsStatus } from '../chat/types';
import { cn } from '@/lib/utils';

interface StandaloneChatPanelProps {
  workflowId: string;
  workflowName: string;
  onWorkflowGraphSync: (graph: unknown) => void;
  onRenameWorkflow: (workflowId: string, newName: string) => Promise<void>;
  onDeleteWorkflow: (workflowId: string) => Promise<void>;
  readOnly?: boolean;
}

// ── Workflow Name Field ──────────────────────────────────────────────────────
interface WorkflowNameFieldProps {
  workflowName: string;
  onRename: (newName: string) => Promise<void>;
  onDeleteClick: () => void;
  readOnly?: boolean;
  readOnlyReason?: string | null;
}

function WorkflowNameField({
  workflowName,
  onRename,
  onDeleteClick,
  readOnly = false,
  readOnlyReason = null,
}: WorkflowNameFieldProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState(workflowName);
  const [isRenaming, setIsRenaming] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  useEffect(() => {
    if (!isEditing) setEditName(workflowName);
  }, [workflowName, isEditing]);

  const cancelEditing = useCallback(() => {
    setEditName(workflowName);
    setIsEditing(false);
  }, [workflowName]);

  const saveEdit = useCallback(async () => {
    const trimmed = editName.trim();
    if (!trimmed || trimmed === workflowName) { cancelEditing(); return; }
    setIsRenaming(true);
    try {
      await onRename(trimmed);
      setIsEditing(false);
    } catch {
      setEditName(workflowName);
    } finally {
      setIsRenaming(false);
    }
  }, [editName, workflowName, onRename, cancelEditing]);

  const startEditing = useCallback(() => {
    if (readOnly) return;
    setEditName(workflowName);
    setIsEditing(true);
  }, [readOnly, workflowName]);
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') { e.preventDefault(); saveEdit(); }
    else if (e.key === 'Escape') { cancelEditing(); }
  };

  if (isEditing) {
    return (
      <div className="flex items-center gap-1.5 flex-1 min-w-0">
        <Input ref={inputRef} value={editName} onChange={(e) => setEditName(e.target.value)}
          onKeyDown={handleKeyDown} className="h-7 text-sm flex-1" disabled={isRenaming} />
        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={saveEdit} disabled={isRenaming} title="Save">
          <Check className="w-3 h-3" />
        </Button>
        <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={cancelEditing} disabled={isRenaming} title="Cancel">
          <X className="w-3 h-3" />
        </Button>
      </div>
    );
  }

  return (
    <>
      <button
        className="flex items-center gap-1.5 h-8 px-1.5 rounded-lg min-w-0 max-w-[220px] hover:bg-accent/60 transition-all duration-150 outline-none focus-visible:ring-1 focus-visible:ring-ring"
        onClick={startEditing}
        title={readOnly ? readOnlyReason ?? 'Workflow is read-only' : 'Click to rename'}
        disabled={readOnly}
      >
        <span className="text-sm font-medium truncate text-foreground">{workflowName}</span>
      </button>
      <div className="flex-1" />
      <div className="flex items-center gap-0.5 shrink-0">
        <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-foreground"
          onClick={startEditing} title={readOnly ? readOnlyReason ?? 'Workflow is read-only' : 'Rename workflow'} disabled={readOnly}>
          <FileEdit className="w-3.5 h-3.5" />
        </Button>
        <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive transition-colors"
          onClick={onDeleteClick} title={readOnly ? readOnlyReason ?? 'Workflow is read-only' : 'Delete workflow'} disabled={readOnly}>
          <Trash2 className="w-3.5 h-3.5" />
        </Button>
      </div>
    </>
  );
}

// ── Workflow Bar ────────────────────────────────────────────────────────────
interface WorkflowBarProps {
  workflowName: string;
  onRename: (newName: string) => Promise<void>;
  onDeleteClick: () => void;
  readOnly?: boolean;
  readOnlyReason?: string | null;
}

function WorkflowBar({
  workflowName,
  onRename,
  onDeleteClick,
  readOnly = false,
  readOnlyReason = null,
}: WorkflowBarProps) {
  return (
    <div className="h-12 px-3 flex items-center gap-2 border-b border-border bg-card shrink-0">
      <WorkflowNameField
        workflowName={workflowName}
        onRename={onRename}
        onDeleteClick={onDeleteClick}
        readOnly={readOnly}
        readOnlyReason={readOnlyReason}
      />
    </div>
  );
}

// ── Chat Bar ─────────────────────────────────────────────────────────────────
interface ChatBarProps {
  onNewSession: () => void;
  sessions: ChatSession[];
  sessionsStatus: SessionsStatus;
  onSelectSession: (sessionId: number) => void;
}

function ChatBar({ onNewSession, sessions, sessionsStatus, onSelectSession }: ChatBarProps) {
  const [sessionPopoverOpen, setSessionPopoverOpen] = useState(false);
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const isLoading = useChatStore((state) => state.isLoading);

  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-border/40 bg-background shrink-0">
      {/* Status indicator + label */}
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'w-1.5 h-1.5 rounded-full shrink-0',
            isLoading ? 'bg-[hsl(var(--seer))] animate-pulse' : 'bg-emerald-400/80',
          )}
          title={isLoading ? 'Thinking...' : 'Ready'}
        />
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Chat</span>
      </div>

      {/* Session controls */}
      <SessionPopover
        open={sessionPopoverOpen}
        onOpenChange={setSessionPopoverOpen}
        sessions={sessions}
        isPending={sessionsStatus.isPending}
        isError={sessionsStatus.isError}
        error={sessionsStatus.error}
        hasNextPage={sessionsStatus.hasNextPage}
        fetchNextPage={sessionsStatus.fetchNextPage}
        isFetchingNextPage={sessionsStatus.isFetchingNextPage}
        currentSessionId={currentSessionId}
        onSelectSession={(id) => {
          onSelectSession(id);
          setSessionPopoverOpen(false);
        }}
        onNewSession={onNewSession}
      />
    </div>
  );
}

export function StandaloneChatPanel({
  workflowId,
  workflowName,
  onWorkflowGraphSync,
  onRenameWorkflow,
  onDeleteWorkflow,
  readOnly = false,
}: StandaloneChatPanelProps) {
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const proposalPreview = useUIStore((state) => state.proposalPreview);
  const setProposalPreview = useUIStore((state) => state.setProposalPreview);

  const pendingAutoSendMessage = useChatStore((s) => s.pendingAutoSendMessage);
  const setPendingAutoSendMessage = useChatStore((s) => s.setPendingAutoSendMessage);

  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const navigate = useNavigate();
  const { handleSend, handleStop, handleResumeMessage, handleAnswerClarification, handleAnswerClarifications, handleNewSession, handleSelectSession } =
    useChatActions(workflowId, nodes, edges);
  useInitialChatMessage(workflowId, handleSend);

  useEffect(() => {
    if (!pendingAutoSendMessage) return;
    setPendingAutoSendMessage(null);
    setTimeout(() => handleSend(), 100);
  }, [pendingAutoSendMessage, setPendingAutoSendMessage, handleSend]);

  const { handleAcceptProposal, handleRejectProposal } = useProposalActions(workflowId, onWorkflowGraphSync);
  const { models, isLoadingModels } = useAvailableModels();
  const { sessions, sessionsQuery } = useChatSessionData(workflowId);

  const handleRename = useCallback(
    (newName: string) => onRenameWorkflow(workflowId, newName),
    [workflowId, onRenameWorkflow],
  );

  const handleDeleteConfirm = useCallback(async () => {
    setIsDeleting(true);
    try {
      await onDeleteWorkflow(workflowId);
      navigate('/');
    } finally {
      setIsDeleting(false);
      setIsDeleteDialogOpen(false);
    }
  }, [workflowId, onDeleteWorkflow, navigate]);

  const sessionsStatus: SessionsStatus = {
    isPending: sessionsQuery.isPending,
    isError: sessionsQuery.isError,
    error: sessionsQuery.error,
    hasNextPage: sessionsQuery.hasNextPage,
    fetchNextPage: sessionsQuery.fetchNextPage,
    isFetchingNextPage: sessionsQuery.isFetchingNextPage,
  };

  return (
    <div className="relative flex h-full min-w-0 w-full flex-col border-r bg-card">
      <WorkflowBar
        workflowName={workflowName}
        onRename={handleRename}
        onDeleteClick={() => setIsDeleteDialogOpen(true)}
        readOnly={readOnly}
        readOnlyReason="You need the edit lock to change workflow metadata."
      />
      <ChatBar
        onNewSession={handleNewSession}
        sessions={sessions}
        sessionsStatus={sessionsStatus}
        onSelectSession={(sessionId: number) => handleSelectSession(sessionId, sessions)}
      />

      <div className="min-w-0 flex-1 overflow-hidden">
        <ChatPanel
          workflowId={workflowId}
          onSend={handleSend}
          onResumeSend={handleResumeMessage}
          onAnswerClarification={handleAnswerClarification}
          onAnswerClarifications={handleAnswerClarifications}
          models={models}
          isLoadingModels={isLoadingModels}
          filterSystemPrompt={filterSystemPrompt}
          onAcceptProposal={(proposalId) => handleAcceptProposal(proposalId).then(() => setProposalPreview(null))}
          onRejectProposal={(proposalId) => handleRejectProposal(proposalId).then(() => setProposalPreview(null))}
          activePreviewProposalId={proposalPreview?.proposal.id ?? null}
          onStop={handleStop}
        />
      </div>

      <DeleteWorkflowDialog
        open={isDeleteDialogOpen}
        onOpenChange={setIsDeleteDialogOpen}
        workflowName={workflowName}
        onConfirm={handleDeleteConfirm}
        isDeleting={isDeleting}
      />
    </div>
  );
}
