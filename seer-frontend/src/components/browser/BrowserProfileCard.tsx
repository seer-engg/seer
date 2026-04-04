/**
 * Browser profiles page — gallery layout with recordings per profile.
 */
import { useState } from 'react';
import { Globe, Plus, Trash2, LogIn, Loader2 } from 'lucide-react';
import { useBrowserProfiles } from '@/hooks/useBrowserProfiles';
import { useToast } from '@/hooks/utility/use-toast';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { BrowserProfileDialog } from './BrowserProfileDialog';
import { BrowserRecordingsList } from './BrowserRecordingsList';
import { SessionReplay } from './SessionReplay';
import type { BrowserProfile } from '@/types/browser';

function ProfileSection({
  profile,
  onLogin,
  onDelete,
  isDeleting,
  onPlayRecording,
}: {
  profile: BrowserProfile;
  onLogin: () => void;
  onDelete: () => void;
  isDeleting: boolean;
  onPlayRecording: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      {/* Profile header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3 min-w-0">
          <h3 className="font-medium text-sm truncate">{profile.name}</h3>
          <div className="flex items-center gap-1 flex-wrap">
            {profile.logged_in_domains.slice(0, 3).map((domain) => (
              <Badge key={domain} variant="secondary" className="h-4 px-1.5 text-[9px]">
                {domain}
              </Badge>
            ))}
            {profile.logged_in_domains.length > 3 && (
              <Badge variant="outline" className="h-4 px-1.5 text-[9px]">
                +{profile.logged_in_domains.length - 3}
              </Badge>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={onLogin} title="Login to services">
            <LogIn className="h-3.5 w-3.5" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-destructive hover:text-destructive"
            onClick={onDelete}
            disabled={isDeleting}
            title="Delete profile"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Recordings gallery */}
      <BrowserRecordingsList profileId={profile.id} limit={8} onPlay={onPlayRecording} />
    </div>
  );
}

export function BrowserProfileCard() {
  const { toast } = useToast();
  const { profiles, isLoading, createProfile, deleteProfile, isCreating, isDeleting } = useBrowserProfiles();

  const [dialogMode, setDialogMode] = useState<'create' | 'login' | null>(null);
  const [selectedProfile, setSelectedProfile] = useState<BrowserProfile | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [profileToDelete, setProfileToDelete] = useState<BrowserProfile | null>(null);
  const [replayRecordingId, setReplayRecordingId] = useState<string | null>(null);

  const handleCreate = async (name: string) => {
    try {
      await createProfile(name);
      toast({ title: 'Profile created', description: `"${name}" has been created.` });
      setDialogMode(null);
    } catch (error) {
      toast({ title: 'Failed to create profile', description: error instanceof Error ? error.message : 'An error occurred', variant: 'destructive' });
    }
  };

  const handleDeleteConfirm = async () => {
    if (!profileToDelete) return;
    try {
      await deleteProfile(profileToDelete.id);
      toast({ title: 'Profile deleted' });
    } catch (error) {
      toast({ title: 'Failed to delete profile', description: error instanceof Error ? error.message : 'An error occurred', variant: 'destructive' });
    } finally {
      setDeleteDialogOpen(false);
      setProfileToDelete(null);
    }
  };

  return (
    <>
      {/* New Profile button */}
      <Button onClick={() => setDialogMode('create')} disabled={isCreating} className="">
        {isCreating ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Plus className="h-4 w-4 mr-2" />}
        New Profile
      </Button>

      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )}

      {!isLoading && profiles.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <Globe className="h-12 w-12 mx-auto mb-3 opacity-50" />
          <p className="text-sm">No browser profiles yet</p>
          <p className="text-xs mt-1">Create a profile and login to save authenticated sessions.</p>
        </div>
      )}

      {!isLoading && profiles.length > 0 && (
        <div className="space-y-8">
          {profiles.map((profile) => (
            <ProfileSection
              key={profile.id}
              profile={profile}
              onLogin={() => { setSelectedProfile(profile); setDialogMode('login'); }}
              onDelete={() => { setProfileToDelete(profile); setDeleteDialogOpen(true); }}
              isDeleting={isDeleting}
              onPlayRecording={setReplayRecordingId}
            />
          ))}
        </div>
      )}

      <BrowserProfileDialog
        mode={dialogMode}
        profile={selectedProfile}
        onClose={() => { setDialogMode(null); setSelectedProfile(null); }}
        onCreate={handleCreate}
        isCreating={isCreating}
      />

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Profile?</AlertDialogTitle>
            <AlertDialogDescription>
              Delete "{profileToDelete?.name}"? All saved login sessions will be removed.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteConfirm} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Replay dialog */}
      <Dialog open={!!replayRecordingId} onOpenChange={(open) => !open && setReplayRecordingId(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>Session Replay</DialogTitle>
          </DialogHeader>
          <SessionReplay recordingId={replayRecordingId} autoPlay />
        </DialogContent>
      </Dialog>
    </>
  );
}
