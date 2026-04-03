import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useFixWithAI } from '@/hooks/useFixWithAI';

export interface FixWithAIProps {
  /** Error text to include in the AI message */
  error: string;
  /** Required for cross-page navigation (executions page) */
  workflowId?: string;
  className?: string;
}

export function FixWithAI({ error, workflowId, className }: FixWithAIProps) {
  const { fixWithAI } = useFixWithAI();

  return (
    <Button
      variant="ghost"
      size="sm"
      className={cn('h-7 px-2 gap-1.5 text-xs text-seer hover:text-seer hover:bg-seer/10', className)}
      onClick={() => fixWithAI(error, workflowId)}
    >
      <Sparkles className="h-3 w-3" />
      Fix with AI
    </Button>
  );
}
