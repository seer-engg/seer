import { Trash2, Plus, Pencil } from 'lucide-react';
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import type { GeneralChatSession } from '@/stores/generalChatStore';

interface ChatSessionListProps {
  sessions: GeneralChatSession[];
  activeSessionId: number | null;
  onSelect: (id: number) => void;
  onDelete: (id: number) => void;
  onCreate: () => void;
  onRename: (id: number, title: string) => void;
}

export function ChatSessionList({ sessions, activeSessionId, onSelect, onDelete, onCreate, onRename }: ChatSessionListProps) {
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editTitle, setEditTitle] = useState('');

  const startRename = (session: GeneralChatSession) => {
    setEditingId(session.id);
    setEditTitle(session.title || '');
  };

  const submitRename = (id: number) => {
    if (editTitle.trim()) {
      onRename(id, editTitle.trim());
    }
    setEditingId(null);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 pt-6">
        <h3 className="text-sm font-semibold mb-3">Chat History</h3>
        <Button onClick={onCreate} variant="outline" size="sm" className="w-full">
          <Plus className="w-4 h-4 mr-2" />
          New Chat
        </Button>
      </div>
      <Separator />
      <ScrollArea className="flex-1">
        <div className="p-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              className={cn(
                'group flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer transition-colors',
                'hover:bg-muted/50',
                activeSessionId === session.id && 'bg-muted',
              )}
              onClick={() => onSelect(session.id)}
            >
              <div className="flex-1 min-w-0">
                {editingId === session.id ? (
                  <input
                    className="w-full text-sm bg-transparent border-b border-primary outline-none"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onBlur={() => submitRename(session.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') submitRename(session.id);
                      if (e.key === 'Escape') setEditingId(null);
                    }}
                    autoFocus
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <p className="text-sm truncate">{session.title || 'New Chat'}</p>
                )}
                <p className="text-xs text-muted-foreground mt-0.5">
                  {new Date(session.updated_at).toLocaleDateString()}
                </p>
              </div>
              <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-foreground transition-colors"
                  onClick={(e) => { e.stopPropagation(); startRename(session); }}
                >
                  <Pencil className="w-3 h-3" />
                </button>
                <button
                  className="p-1 rounded hover:bg-background text-muted-foreground hover:text-destructive transition-colors"
                  onClick={(e) => { e.stopPropagation(); onDelete(session.id); }}
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-sm text-muted-foreground p-4 text-center">No conversations yet</p>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
