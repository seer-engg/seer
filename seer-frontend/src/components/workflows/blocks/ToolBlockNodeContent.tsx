import { Loader2, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ToolBlockNodeContentProps {
  icon: React.ReactNode;
  isLoading: boolean;
  needsAuth: boolean;
  label: string;
  statusBadge: React.ReactNode;
  handleConnect: (e: React.MouseEvent) => void;
}

export function ToolBlockNodeContent({
  icon,
  isLoading,
  needsAuth,
  label,
  statusBadge,
  handleConnect,
}: ToolBlockNodeContentProps) {
  return (
    <div className="space-y-2">
      {/* Icon, tool name, and status row */}
      <div className="flex items-center gap-1">
        <div
          className={cn(
            'w-8 h-8 rounded flex items-center justify-center shrink-0',
            needsAuth ? 'bg-amber-500/10' : 'bg-primary/10',
          )}
        >
          {isLoading ? (
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
          ) : (
            <div className={cn(needsAuth ? 'text-amber-600 dark:text-amber-400' : 'text-primary')}>
              {icon}
            </div>
          )}
        </div>
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{label}</p>
        </div>
        {statusBadge && (
          <div className="flex items-center gap-2">
            <div className="shrink-0">{statusBadge}</div>
            {needsAuth && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 text-xs text-amber-600 dark:text-amber-400 hover:bg-amber-500/10 shrink-0"
                onClick={handleConnect}
              >
                Connect
                <ExternalLink className="w-3 h-3 ml-1" />
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
