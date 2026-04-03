/**
 * Connection status indicator for browser viewer.
 */
import { Loader2, CheckCircle, AlertCircle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { StreamStatus } from '@/types/browser';

interface BrowserViewerStatusProps {
  status: StreamStatus;
  error?: string | null;
  className?: string;
}

const statusConfig: Record<
  StreamStatus,
  { icon: typeof Loader2; label: string; className: string }
> = {
  connecting: {
    icon: Loader2,
    label: 'Connecting...',
    className: 'text-amber-600 dark:text-amber-400',
  },
  connected: {
    icon: CheckCircle,
    label: 'Connected',
    className: 'text-emerald-600 dark:text-emerald-400',
  },
  navigating: {
    icon: Loader2,
    label: 'Navigating...',
    className: 'text-blue-600 dark:text-blue-400',
  },
  error: {
    icon: AlertCircle,
    label: 'Error',
    className: 'text-destructive',
  },
  closed: {
    icon: XCircle,
    label: 'Disconnected',
    className: 'text-muted-foreground',
  },
};

export function BrowserViewerStatus({
  status,
  error,
  className,
}: BrowserViewerStatusProps) {
  const config = statusConfig[status];
  const Icon = config.icon;
  const isAnimated = status === 'connecting' || status === 'navigating';

  return (
    <div className={cn('flex items-center gap-1.5 text-xs', className)}>
      <Icon
        className={cn(
          'h-3.5 w-3.5',
          config.className,
          isAnimated && 'animate-spin'
        )}
      />
      <span className={config.className}>
        {error || config.label}
      </span>
    </div>
  );
}
