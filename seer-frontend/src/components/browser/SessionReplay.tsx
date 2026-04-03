/**
 * rrweb session replay component.
 *
 * Wraps rrweb-player to replay recorded browser sessions.
 */
import { useRef, useEffect, useState } from 'react';
import { Loader2, AlertCircle, Play, Clock, FileJson } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBrowserReplay } from '@/hooks/useBrowserReplay';
import { Badge } from '@/components/ui/badge';

interface SessionReplayProps {
  recordingId: string | null;
  className?: string;
  autoPlay?: boolean;
  publicMode?: boolean;
}

function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function SessionReplayLoading({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center justify-center h-[400px] bg-muted/50 rounded-lg', className)}>
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Loading recording...</span>
      </div>
    </div>
  );
}

function SessionReplayError({ className, message }: { className?: string; message?: string }) {
  return (
    <div className={cn('flex items-center justify-center h-[400px] bg-destructive/10 rounded-lg', className)}>
      <div className="flex flex-col items-center gap-2 text-destructive">
        <AlertCircle className="h-8 w-8" />
        <span className="text-sm font-medium">Failed to load recording</span>
        <span className="text-xs">{message}</span>
      </div>
    </div>
  );
}

function NoRecordingSelected({ className }: { className?: string }) {
  return (
    <div className={cn('flex items-center justify-center h-[400px] bg-muted/50 rounded-lg', className)}>
      <div className="flex flex-col items-center gap-2 text-muted-foreground">
        <Play className="h-8 w-8" />
        <span className="text-sm">Select a recording to replay</span>
      </div>
    </div>
  );
}

export function SessionReplay({
  recordingId,
  className,
  autoPlay = false,
  publicMode = false,
}: SessionReplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<unknown>(null);
  const [playerLoaded, setPlayerLoaded] = useState(false);
  const [playerError, setPlayerError] = useState<string | null>(null);

  const { metadata, events, isLoading, error, isReady } = useBrowserReplay(recordingId, { public: publicMode });

  // Dynamically load rrweb-player
  useEffect(() => {
    if (!isReady || !containerRef.current || events.length === 0) return;

    let mounted = true;
    const container = containerRef.current;

    const loadPlayer = async () => {
      try {
        // Dynamic import of rrweb-player
        const rrwebPlayer = await import('rrweb-player');
        await import('rrweb-player/dist/style.css');

        if (!mounted || !container) return;

        // Create new player
        const player = new rrwebPlayer.default({
          target: container,
          props: {
            events: events as unknown as never[], // rrweb-player types expect eventWithTime[]
            autoPlay: autoPlay,
            showController: true,
            width: 800,
            height: 600,
          },
        });

        playerRef.current = player;
        setPlayerLoaded(true);
        setPlayerError(null);
      } catch (err) {
        console.error('Failed to load rrweb-player:', err);
        setPlayerError('Failed to load replay player');
      }
    };

    loadPlayer();

    return () => {
      mounted = false;
      // Clean up rrweb-player DOM before React tries to reconcile
      if (container) {
        container.innerHTML = '';
      }
      playerRef.current = null;
      setPlayerLoaded(false);
    };
  }, [isReady, events, autoPlay]);

  if (isLoading) return <SessionReplayLoading className={className} />;
  if (error || playerError) return <SessionReplayError className={className} message={error?.message || playerError || undefined} />;
  if (!recordingId) return <NoRecordingSelected className={className} />;

  return (
    <div className={cn('flex flex-col gap-3', className)}>
      {/* Metadata header */}
      {metadata && (
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <Badge variant="secondary" className="h-5">
            {metadata.session_type}
          </Badge>
          <div className="flex items-center gap-1">
            <Clock className="h-3 w-3" />
            <span>{formatDuration(metadata.duration_ms)}</span>
          </div>
          <div className="flex items-center gap-1">
            <FileJson className="h-3 w-3" />
            <span>{metadata.event_count} events</span>
          </div>
          <span>{formatFileSize(metadata.compressed_size_bytes)}</span>
          {metadata.start_url && (
            <span className="truncate max-w-[200px]" title={metadata.start_url}>
              {metadata.start_url}
            </span>
          )}
        </div>
      )}

      {/* Player wrapper - keeps loading indicator separate from rrweb-player container */}
      <div className="relative rounded-lg border overflow-hidden bg-black min-h-[400px]">
        {/* Loading overlay - rendered by React, separate from player container */}
        {!playerLoaded && isReady && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="flex flex-col items-center gap-2">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Initializing player...</span>
            </div>
          </div>
        )}
        {/* rrweb-player container - React should not render children here */}
        <div ref={containerRef} className="w-full h-full" />
      </div>
    </div>
  );
}
