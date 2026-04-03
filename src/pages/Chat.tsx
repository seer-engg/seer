import { useCallback, useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { History } from 'lucide-react';
import { useGeneralChatStore } from '@/stores/generalChatStore';
import { useGeneralChat } from '@/hooks/useGeneralChat';
import { ChatSessionList } from '@/components/chat/ChatSessionList';
import { GeneralChatView } from '@/components/chat/GeneralChatView';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from '@/components/ui/sheet';

export default function Chat() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const navigate = useNavigate();
  const sessions = useGeneralChatStore((s) => s.sessions);
  const activeSessionId = useGeneralChatStore((s) => s.activeSessionId);
  const { loadSessions, selectSession, createSession, removeSession, renameSession, handleSend, handleStop } = useGeneralChat();
  const [sheetOpen, setSheetOpen] = useState(false);

  const handleCreateSession = useCallback(async () => {
    const session = await createSession();
    navigate(`/chat/${session.id}`);
    setSheetOpen(false);
  }, [createSession, navigate]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (sessionId) {
      const id = parseInt(sessionId, 10);
      if (!isNaN(id) && id !== activeSessionId) {
        selectSession(id);
      }
    } else if (!activeSessionId) {
      handleCreateSession();
    }
  }, [sessionId, activeSessionId, selectSession, handleCreateSession]);

  const handleSelectSession = (id: number) => {
    selectSession(id);
    navigate(`/chat/${id}`);
    setSheetOpen(false);
  };

  const handleDeleteSession = async (id: number) => {
    await removeSession(id);
    if (activeSessionId === id) {
      navigate('/chat');
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Top bar */}
      <div className="flex items-center gap-2 px-4 py-2 border-b bg-background/80 backdrop-blur-sm">
        <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <History className="h-4 w-4" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-72 p-0">
            <ChatSessionList
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSelect={handleSelectSession}
              onDelete={handleDeleteSession}
              onCreate={handleCreateSession}
              onRename={renameSession}
            />
          </SheetContent>
        </Sheet>
        <span className="text-sm font-medium text-muted-foreground">Chat</span>
      </div>

      {/* Main chat area */}
      <div className="flex-1 min-h-0">
        {activeSessionId ? (
          <GeneralChatView onSend={handleSend} onStop={handleStop} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="mb-6">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-seer-500 to-seer-700 flex items-center justify-center mx-auto mb-4">
                <span className="text-white text-lg font-bold">S</span>
              </div>
              <h2 className="text-xl font-semibold mb-1">Welcome to Seer Chat</h2>
              <p className="text-sm text-muted-foreground">Start a new conversation or pick one from history</p>
            </div>
            <Button onClick={handleCreateSession} variant="brand">
              New Conversation
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
