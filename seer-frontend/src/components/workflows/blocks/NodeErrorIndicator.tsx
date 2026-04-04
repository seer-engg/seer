import { memo } from 'react';
import { AlertCircle } from 'lucide-react';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';
import type { NodeErrorState } from '@/stores/canvasStore';

interface NodeErrorIndicatorProps {
  error: NodeErrorState;
}

export const NodeErrorIndicator = memo(function NodeErrorIndicator({ error }: NodeErrorIndicatorProps) {
  return (
    <Tooltip delayDuration={0}>
      <TooltipTrigger asChild>
        <div className="absolute -top-2 -left-2 w-5 h-5 rounded-full bg-bug flex items-center justify-center z-10 cursor-help">
          <AlertCircle className="w-3 h-3 text-white" />
        </div>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-[300px]">
        <div className="space-y-1">
          <p className="font-medium text-sm text-bug">{error.code.replace(/_/g, ' ')}</p>
          <p className="text-xs">{error.message}</p>
          {error.expression && (
            <code className="text-xs font-mono bg-muted px-1 py-0.5 rounded block truncate">
              {error.expression}
            </code>
          )}
        </div>
      </TooltipContent>
    </Tooltip>
  );
});
