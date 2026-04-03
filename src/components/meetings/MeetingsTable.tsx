import { formatDistanceToNow } from 'date-fns';
import { Video, Check, X, ExternalLink } from 'lucide-react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { cn } from '@/lib/utils';
import type { MeetingListItem } from '@/types/meetings';

interface MeetingsTableProps {
  meetings: MeetingListItem[];
  isLoading: boolean;
  onSelect: (botId: string) => void;
}

const STATE_STYLES: Record<string, { label: string; variant: 'default' | 'secondary' | 'outline' | 'destructive'; className?: string }> = {
  ended: { label: 'Ended', variant: 'secondary', className: 'bg-green-500/10 text-green-600 border-green-500/20' },
  joining: { label: 'Joining', variant: 'secondary', className: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20' },
  joined_recording: { label: 'Recording', variant: 'secondary', className: 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20' },
  joined_not_recording: { label: 'In Meeting', variant: 'secondary', className: 'bg-blue-500/10 text-blue-600 border-blue-500/20' },
  leaving: { label: 'Leaving', variant: 'secondary', className: 'bg-gray-500/10 text-gray-600 border-gray-500/20' },
  post_processing: { label: 'Processing', variant: 'secondary', className: 'bg-purple-500/10 text-purple-600 border-purple-500/20' },
};

function StateBadge({ state }: { state: string }) {
  const style = STATE_STYLES[state] ?? { label: state, variant: 'outline' as const };
  return (
    <Badge variant={style.variant} className={cn('text-xs', style.className)}>
      {style.label}
    </Badge>
  );
}

function StatusIcon({ complete }: { complete: boolean }) {
  return complete ? (
    <Check className="h-4 w-4 text-green-500" />
  ) : (
    <X className="h-4 w-4 text-muted-foreground/40" />
  );
}

function truncateUrl(url: string, maxLen = 40): string {
  try {
    const parsed = new URL(url);
    const display = parsed.hostname + parsed.pathname;
    return display.length > maxLen ? display.slice(0, maxLen) + '...' : display;
  } catch {
    return url.length > maxLen ? url.slice(0, maxLen) + '...' : url;
  }
}

function LoadingSkeleton() {
  return (
    <>
      {[1, 2, 3, 4, 5].map((i) => (
        <TableRow key={i}>
          <TableCell><Skeleton className="h-5 w-20" /></TableCell>
          <TableCell><Skeleton className="h-4 w-48" /></TableCell>
          <TableCell><Skeleton className="h-4 w-24" /></TableCell>
          <TableCell><Skeleton className="h-4 w-4 mx-auto" /></TableCell>
          <TableCell><Skeleton className="h-4 w-4 mx-auto" /></TableCell>
        </TableRow>
      ))}
    </>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="p-3 rounded-full bg-violet-500/10 mb-4">
        <Video className="h-8 w-8 text-violet-500" />
      </div>
      <h3 className="text-lg font-medium mb-1">No meetings yet</h3>
      <p className="text-sm text-muted-foreground max-w-sm">
        Meetings will appear here when your workflows send a meetingsEA bot to record a meeting.
      </p>
    </div>
  );
}

export function MeetingsTable({ meetings, isLoading, onSelect }: MeetingsTableProps) {
  if (!isLoading && meetings.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[120px]">Status</TableHead>
            <TableHead>Meeting</TableHead>
            <TableHead className="w-[160px]">Created</TableHead>
            <TableHead className="w-[100px] text-center">Transcript</TableHead>
            <TableHead className="w-[100px] text-center">Recording</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {isLoading ? (
            <LoadingSkeleton />
          ) : (
            meetings.map((meeting) => (
              <TableRow
                key={meeting.bot_id}
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => onSelect(meeting.bot_id)}
              >
                <TableCell>
                  <StateBadge state={meeting.state} />
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="text-sm truncate">
                      {meeting.meeting_url ? truncateUrl(meeting.meeting_url) : 'Unknown meeting'}
                    </span>
                    {meeting.meeting_url && (
                      <a
                        href={meeting.meeting_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="text-muted-foreground hover:text-foreground opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <ExternalLink className="h-3.5 w-3.5" />
                      </a>
                    )}
                  </div>
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {meeting.created_at
                    ? formatDistanceToNow(new Date(meeting.created_at), { addSuffix: true })
                    : '-'}
                </TableCell>
                <TableCell className="text-center">
                  <StatusIcon complete={meeting.transcription_state === 'complete'} />
                </TableCell>
                <TableCell className="text-center">
                  <StatusIcon complete={meeting.recording_state === 'complete'} />
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
