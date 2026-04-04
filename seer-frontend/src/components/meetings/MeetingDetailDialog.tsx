import { ExternalLink, Video } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { useMeetingTranscript, useMeetingRecording } from '@/hooks/useMeetings';
import type { MeetingListItem } from '@/types/meetings';

interface MeetingDetailDialogProps {
  meeting: MeetingListItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function formatTimestamp(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function TranscriptView({ botId }: { botId: string }) {
  const { utterances, isLoading } = useMeetingTranscript(botId);

  if (isLoading) {
    return (
      <div className="space-y-4 p-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="space-y-1">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-4 w-full" />
          </div>
        ))}
      </div>
    );
  }

  if (utterances.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <p className="text-sm text-muted-foreground">No transcript available</p>
      </div>
    );
  }

  return (
    <div className="space-y-3 p-4 max-h-[60vh] overflow-y-auto">
      {utterances.map((u, idx) => (
        <div key={idx} className="flex gap-3">
          <span className="text-xs text-muted-foreground font-mono min-w-[45px] pt-0.5">
            {formatTimestamp(u.start_ms)}
          </span>
          <div className="flex-1 min-w-0">
            <span className="text-sm font-medium">{u.speaker}</span>
            <p className="text-sm text-muted-foreground">{u.text}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

function RecordingView({ botId }: { botId: string }) {
  const { recording, isLoading } = useMeetingRecording(botId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Skeleton className="h-48 w-full max-w-md rounded-lg" />
      </div>
    );
  }

  if (!recording?.url) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <Video className="h-8 w-8 text-muted-foreground/40 mb-2" />
        <p className="text-sm text-muted-foreground">Recording not available</p>
      </div>
    );
  }

  return (
    <div className="p-4">
      <video
        controls
        className="w-full rounded-lg bg-black"
        src={recording.url}
      >
        Your browser does not support the video tag.
      </video>
    </div>
  );
}

export function MeetingDetailDialog({ meeting, open, onOpenChange }: MeetingDetailDialogProps) {
  if (!meeting) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-3">
            <div className="p-1.5 rounded-md bg-violet-500/10">
              <Video className="h-4 w-4 text-violet-500" />
            </div>
            <span className="truncate">
              {meeting.meeting_url ? new URL(meeting.meeting_url).hostname : 'Meeting'}
            </span>
            <Badge variant="secondary" className="text-xs ml-auto">
              {meeting.state}
            </Badge>
          </DialogTitle>
        </DialogHeader>

        <div className="flex items-center gap-4 text-xs text-muted-foreground px-1">
          {meeting.meeting_url && (
            <a
              href={meeting.meeting_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-foreground transition-colors"
            >
              Meeting link <ExternalLink className="h-3 w-3" />
            </a>
          )}
          {meeting.created_at && (
            <span>
              {formatDistanceToNow(new Date(meeting.created_at), { addSuffix: true })}
            </span>
          )}
          <span className="font-mono text-[10px]">{meeting.bot_id}</span>
        </div>

        <Tabs defaultValue="transcript" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="mx-1">
            <TabsTrigger value="transcript">Transcript</TabsTrigger>
            <TabsTrigger value="recording">Recording</TabsTrigger>
          </TabsList>
          <TabsContent value="transcript" className="flex-1 overflow-hidden mt-0">
            <TranscriptView botId={meeting.bot_id} />
          </TabsContent>
          <TabsContent value="recording" className="flex-1 overflow-hidden mt-0">
            <RecordingView botId={meeting.bot_id} />
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}
