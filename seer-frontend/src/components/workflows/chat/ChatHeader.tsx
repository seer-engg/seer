import { SessionPopover } from './SessionPopover';
import type { ChatSession } from '../buildtypes';
import type { SessionsStatus } from './types';
import { useChatStore } from '@/stores';

interface ChatHeaderProps {
  onNewSession: () => void;
  sessionPopoverOpen: boolean;
  onSessionPopoverOpenChange: (open: boolean) => void;
  sessions: ChatSession[];
  sessionsStatus: SessionsStatus;
  onSelectSession: (sessionId: number) => void;
}

export function ChatHeader({
  onNewSession,
  sessionPopoverOpen,
  onSessionPopoverOpenChange,
  sessions,
  sessionsStatus,
  onSelectSession,
}: ChatHeaderProps) {
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const isLoading = useChatStore((state) => state.isLoading);

  return (
    <div className="px-4 py-2.5 flex items-center justify-between border-b border-border/60 flex-shrink-0">
      <div className="flex items-center gap-2">
        <div className="relative flex items-center justify-center">
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              isLoading
                ? 'bg-[hsl(var(--seer))] animate-pulse'
                : 'bg-emerald-400/80'
            }`}
          />
        </div>
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Chat
        </span>
      </div>

      <SessionPopover
        open={sessionPopoverOpen}
        onOpenChange={onSessionPopoverOpenChange}
        sessions={sessions}
        isPending={sessionsStatus.isPending}
        isError={sessionsStatus.isError}
        error={sessionsStatus.error}
        hasNextPage={sessionsStatus.hasNextPage}
        fetchNextPage={sessionsStatus.fetchNextPage}
        isFetchingNextPage={sessionsStatus.isFetchingNextPage}
        currentSessionId={currentSessionId}
        onSelectSession={onSelectSession}
        onNewSession={onNewSession}
      />
    </div>
  );
}
