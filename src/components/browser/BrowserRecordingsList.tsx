/**
 * Grid gallery of browser session recordings with thumbnails.
 */
import { useState } from 'react';
import { Trash2, Loader2, Video, Clock, Share2 } from 'lucide-react';
import { useBrowserRecordings } from '@/hooks/useBrowserRecordings';
import { useToast } from '@/hooks/utility/use-toast';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
import { RecordingThumbnail } from './RecordingThumbnail';
import type { RecordingMetadata } from '@/types/browser';

interface BrowserRecordingsListProps {
  profileId?: string;
  workflowRunId?: string;
  limit?: number;
  onPlay?: (recordingId: string) => void;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function RecordingCard({
  recording,
  onPlay,
  onShare,
  onDelete,
  isDeleting,
}: {
  recording: RecordingMetadata;
  onPlay: () => void;
  onShare: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}) {
  return (
    <div
      className="group rounded-lg border bg-card overflow-hidden cursor-pointer hover:border-foreground/20 transition-colors"
      onClick={onPlay}
    >
      <RecordingThumbnail recordingId={recording.id} />
      <div className="p-2 space-y-1">
        <div className="flex items-center gap-1.5">
          <Badge variant="secondary" className="h-4 px-1.5 text-[9px]">
            {recording.session_type}
          </Badge>
          <span className="text-[10px] text-muted-foreground">{formatDate(recording.created_at)}</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <span className="flex items-center gap-0.5">
              <Clock className="h-2.5 w-2.5" />
              {formatDuration(recording.duration_ms)}
            </span>
            <span>{recording.event_count} events</span>
          </div>
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={(e) => { e.stopPropagation(); onShare(); }}
              title="Copy share link"
            >
              <Share2 className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-destructive hover:text-destructive"
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              disabled={isDeleting}
              title="Delete"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function BrowserRecordingsList({ profileId, workflowRunId, limit = 10, onPlay }: BrowserRecordingsListProps) {
  const { toast } = useToast();
  const { recordings, total, isLoading, deleteRecording, isDeleting } = useBrowserRecordings({ profileId, workflowRunId, limit });

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [recordingToDelete, setRecordingToDelete] = useState<RecordingMetadata | null>(null);

  const handleShare = async (recording: RecordingMetadata) => {
    const url = `${window.location.origin}/replay/${recording.id}`;
    await navigator.clipboard.writeText(url);
    toast({ title: 'Link copied to clipboard' });
  };

  const handleDeleteConfirm = async () => {
    if (!recordingToDelete) return;
    try {
      await deleteRecording(recordingToDelete.id);
      toast({ title: 'Recording deleted' });
    } catch (error) {
      toast({ title: 'Failed to delete recording', description: error instanceof Error ? error.message : 'An error occurred', variant: 'destructive' });
    } finally {
      setDeleteDialogOpen(false);
      setRecordingToDelete(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (recordings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
        <Video className="h-8 w-8 mb-2 opacity-40" />
        <p className="text-xs">No recordings yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {recordings.map((recording) => (
          <RecordingCard
            key={recording.id}
            recording={recording}
            onPlay={() => onPlay?.(recording.id)}
            onShare={() => handleShare(recording)}
            onDelete={() => { setRecordingToDelete(recording); setDeleteDialogOpen(true); }}
            isDeleting={isDeleting}
          />
        ))}
      </div>
      {total > recordings.length && (
        <p className="text-xs text-muted-foreground text-center">Showing {recordings.length} of {total}</p>
      )}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Recording?</AlertDialogTitle>
            <AlertDialogDescription>This action cannot be undone.</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteConfirm} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
