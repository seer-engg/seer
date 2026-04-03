/**
 * Dialog for creating browser profiles and interactive login.
 *
 * Two modes:
 * - create: Form to create a new profile
 * - login: Embedded BrowserViewer for interactive login
 */
import { useState, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { BrowserViewer } from './BrowserViewer';
import { createBrowserSession, completeBrowserSession } from '@/lib/browser-api';
import { useBrowserStore } from '@/stores/browserStore';
import { useToast } from '@/hooks/utility/use-toast';
import type { BrowserProfile } from '@/types/browser';

interface CreateModeProps {
  profileName: string;
  onProfileNameChange: (name: string) => void;
  onSubmit: (e: React.FormEvent) => void;
  onClose: () => void;
  isCreating: boolean;
}

function CreateModeContent({ profileName, onProfileNameChange, onSubmit, onClose, isCreating }: CreateModeProps) {
  return (
    <>
      <DialogHeader>
        <DialogTitle>Create Browser Profile</DialogTitle>
        <DialogDescription>Create a new browser profile to save authenticated sessions.</DialogDescription>
      </DialogHeader>
      <form onSubmit={onSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="profile-name">Profile Name</Label>
          <Input
            id="profile-name"
            value={profileName}
            onChange={(e) => onProfileNameChange(e.target.value)}
            placeholder="e.g., Work Profile, Personal"
            autoFocus
          />
        </div>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>Cancel</Button>
          <Button type="submit" disabled={!profileName.trim() || isCreating}>
            {isCreating && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Create
          </Button>
        </DialogFooter>
      </form>
    </>
  );
}

interface LoginModeProps {
  profile: BrowserProfile;
  sessionId: string | null;
  onStartLogin: () => void;
  onCompleteLogin: () => void;
  onClose: () => void;
  isStartingSession: boolean;
  isCompletingSession: boolean;
}

function LoginModeContent({
  profile, sessionId, onStartLogin, onCompleteLogin, onClose, isStartingSession, isCompletingSession,
}: LoginModeProps) {
  return (
    <>
      <DialogHeader>
        <DialogTitle>Login - {profile.name}</DialogTitle>
        <DialogDescription>Use the browser below to log into services. Click "Save & Close" when done.</DialogDescription>
      </DialogHeader>
      <div className="flex-1 min-h-0">
        {!sessionId ? (
          <div className="flex items-center justify-center h-full">
            <Button onClick={onStartLogin} disabled={isStartingSession}>
              {isStartingSession && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Start Browser Session
            </Button>
          </div>
        ) : (
          <BrowserViewer sessionId={sessionId} className="h-full" showToolbar={true} />
        )}
      </div>
      <DialogFooter>
        <Button onClick={onCompleteLogin} disabled={!sessionId || isCompletingSession}>
          {isCompletingSession && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save & Close
        </Button>
      </DialogFooter>
    </>
  );
}

interface BrowserProfileDialogProps {
  mode: 'create' | 'login' | null;
  profile: BrowserProfile | null;
  onClose: () => void;
  onCreate: (name: string) => Promise<void>;
  isCreating: boolean;
}

export function BrowserProfileDialog({ mode, profile, onClose, onCreate, isCreating }: BrowserProfileDialogProps) {
  const { toast } = useToast();
  const { session, setSession, clearSession } = useBrowserStore();
  const [profileName, setProfileName] = useState('');
  const [isStartingSession, setIsStartingSession] = useState(false);
  const [isCompletingSession, setIsCompletingSession] = useState(false);

  const handleOpenChange = useCallback((open: boolean) => {
    if (!open) { setProfileName(''); clearSession(); onClose(); }
  }, [onClose, clearSession]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!profileName.trim()) return;
    await onCreate(profileName.trim());
    setProfileName('');
  };

  const handleStartLogin = useCallback(async () => {
    if (!profile) return;
    setIsStartingSession(true);
    try {
      const result = await createBrowserSession({ profileId: profile.id, targetUrl: undefined });
      setSession({
        sessionId: result.session_id, profileId: result.profile_id, wsUrl: result.ws_url,
        recordingId: result.recording_id, status: 'connecting', isLoading: false, error: null,
      });
    } catch (error) {
      toast({ title: 'Failed to start session', description: error instanceof Error ? error.message : 'An error occurred', variant: 'destructive' });
    } finally {
      setIsStartingSession(false);
    }
  }, [profile, setSession, toast]);

  const handleCompleteLogin = useCallback(async () => {
    if (!session.sessionId) return;
    setIsCompletingSession(true);
    try {
      const result = await completeBrowserSession(session.sessionId);
      toast({ title: 'Login saved', description: `Saved sessions for: ${result.logged_in_domains.join(', ') || 'no domains'}` });
      clearSession();
      onClose();
    } catch (error) {
      toast({ title: 'Failed to save session', description: error instanceof Error ? error.message : 'An error occurred', variant: 'destructive' });
    } finally {
      setIsCompletingSession(false);
    }
  }, [session.sessionId, clearSession, onClose, toast]);

  return (
    <Dialog open={mode !== null} onOpenChange={handleOpenChange}>
      <DialogContent className={mode === 'login' ? 'max-w-4xl h-[80vh] flex flex-col' : 'max-w-md'} hideCloseButton>
        {mode === 'create' && (
          <CreateModeContent profileName={profileName} onProfileNameChange={setProfileName} onSubmit={handleCreate} onClose={onClose} isCreating={isCreating} />
        )}
        {mode === 'login' && profile && (
          <LoginModeContent
            profile={profile} sessionId={session.sessionId} onStartLogin={handleStartLogin}
            onCompleteLogin={handleCompleteLogin} onClose={onClose} isStartingSession={isStartingSession} isCompletingSession={isCompletingSession}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
