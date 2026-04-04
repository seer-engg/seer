import { Clock, Plus } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

import type { ChatSession } from '../buildtypes';

interface SessionPopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  sessions: ChatSession[];
  isPending: boolean;
  isError: boolean;
  error: unknown;
  hasNextPage?: boolean;
  fetchNextPage: () => void;
  isFetchingNextPage: boolean;
  currentSessionId: number | null;
  onSelectSession: (sessionId: number) => void;
  onNewSession: () => void;
}

export function SessionPopover({
  open,
  onOpenChange,
  sessions,
  isPending,
  isError,
  error,
  hasNextPage,
  fetchNextPage,
  isFetchingNextPage,
  currentSessionId,
  onSelectSession,
  onNewSession,
}: SessionPopoverProps) {
  return (
    <div className="flex items-center gap-1.5">
      <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={onNewSession}>
        <Plus className="w-3 h-3 mr-1" />
        New
      </Button>
      <Popover open={open} onOpenChange={onOpenChange}>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
            <Clock className="w-3 h-3 mr-1" />
            History
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-64 p-2" align="end">
          <div className="space-y-2">
            <div className="text-xs font-medium px-2 py-1.5 text-muted-foreground">Select Session</div>
            <div className="max-h-[300px] overflow-y-auto">
              {isPending ? (
                <div className="text-xs text-muted-foreground px-2 py-1.5">Loading sessions...</div>
              ) : isError ? (
                <div className="text-xs text-destructive px-2 py-1.5">
                  {error instanceof Error ? error.message : 'Error loading sessions'}
                </div>
              ) : sessions.length === 0 ? (
                <div className="text-xs text-muted-foreground px-2 py-1.5">No sessions available</div>
              ) : (
                <>
                  {sessions.map((session) => (
                    <button
                      key={session.id}
                      onClick={() => onSelectSession(session.id)}
                      className={`w-full text-left text-xs px-2 py-1.5 rounded hover:bg-accent ${
                        currentSessionId === session.id ? 'bg-accent' : ''
                      }`}
                    >
                      {session.title || `Session ${session.id}`}
                    </button>
                  ))}
                  {hasNextPage && (
                    <button
                      onClick={() => fetchNextPage()}
                      disabled={isFetchingNextPage}
                      className="w-full text-xs px-2 py-1.5 text-muted-foreground hover:text-foreground hover:bg-accent rounded disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isFetchingNextPage ? 'Loading...' : 'Load More'}
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
}

