/**
 * Renders a mini rrweb thumbnail by replaying the first snapshot at t=0.
 */
import { useRef, useEffect, useState } from 'react';
import { Play, Globe } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBrowserRecordingEvents } from '@/hooks/useBrowserRecordings';

interface RecordingThumbnailProps {
  recordingId: string;
  className?: string;
}

export function RecordingThumbnail({ recordingId, className }: RecordingThumbnailProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loaded, setLoaded] = useState(false);
  const { data, isLoading } = useBrowserRecordingEvents(recordingId);

  useEffect(() => {
    if (!data?.events?.length || !containerRef.current) return;

    let mounted = true;
    const container = containerRef.current;

    const loadPlayer = async () => {
      try {
        const rrwebPlayer = await import('rrweb-player');
        await import('rrweb-player/dist/style.css');

        if (!mounted || !container) return;

        // Render at t=0 paused — just need the first frame
        new rrwebPlayer.default({
          target: container,
          props: {
            events: data.events as unknown as never[],
            autoPlay: false,
            showController: false,
            width: 480,
            height: 300,
          },
        });

        const wrapper = container.querySelector('.replayer-wrapper') as HTMLElement;
        if (wrapper) {
          const scale = container.clientWidth / 1280;
          wrapper.style.transform = `scale(${scale})`;
          wrapper.style.transformOrigin = 'top left';
        }

        setLoaded(true);
      } catch (err) {
        console.error('[RecordingThumbnail] Failed to load rrweb player:', err);
      }
    };

    loadPlayer();

    return () => {
      mounted = false;
      if (container) container.innerHTML = '';
      setLoaded(false);
    };
  }, [data]);

  return (
    <div className={cn(
      'relative w-full aspect-video bg-muted/50 rounded-md overflow-hidden',
      className
    )}>
      {/* rrweb player container */}
      <div
        ref={containerRef}
        className={cn(
          'absolute inset-0 pointer-events-none overflow-hidden [&_.rr-player]:!border-none',
          !loaded && 'opacity-0'
        )}
      />

      {/* Placeholder when not loaded */}
      {!loaded && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground">
          {isLoading ? (
            <div className="w-4 h-4 border-2 border-muted-foreground/30 border-t-muted-foreground rounded-full animate-spin" />
          ) : (
            <Globe className="h-6 w-6 opacity-30" />
          )}
        </div>
      )}

      {/* Play overlay */}
      <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity bg-black/30">
        <Play className="h-8 w-8 text-white fill-white" />
      </div>
    </div>
  );
}
