import { Button } from '@/components/ui/button';
import { AlertCircle } from 'lucide-react';

export interface ChatCostCapErrorProps {
  accumulatedCost: number;
  costCap: number;
  onIncreaseLimitClick: () => void;
  onStartFreshClick: () => void;
}

/**
 * Display component for chat cost cap exceeded errors.
 * Shows clear explanation, cost details, and actionable options.
 */
export function ChatCostCapError({
  accumulatedCost,
  costCap,
  onIncreaseLimitClick,
  onStartFreshClick,
}: ChatCostCapErrorProps) {
  return (
    <div className="rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
      {/* Header */}
      <div className="flex items-start gap-3">
        <AlertCircle className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h4 className="font-semibold text-red-900 dark:text-red-100">
            Chat Cost Limit Exceeded
          </h4>
          <p className="mt-1 text-sm text-red-800 dark:text-red-200">
            This chat exceeded your per-conversation cost limit.
          </p>

          {/* Cost Details */}
          <div className="mt-3 rounded-md bg-red-100 dark:bg-red-900/50 p-3">
            <div className="text-sm font-mono">
              <span className="text-red-900 dark:text-red-100">
                ${accumulatedCost.toFixed(2)}
              </span>
              <span className="text-red-700 dark:text-red-300 mx-2">/</span>
              <span className="text-red-700 dark:text-red-300">
                ${costCap.toFixed(2)} limit
              </span>
            </div>
          </div>

          {/* Explanation */}
          <p className="mt-3 text-sm text-red-800 dark:text-red-200">
            You can either increase your cost limit in settings or start a fresh conversation.
          </p>

          {/* Action Buttons */}
          <div className="mt-4 flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onIncreaseLimitClick}
              className="border-red-300 hover:bg-red-100 dark:border-red-700 dark:hover:bg-red-900"
            >
              Increase Limit
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onStartFreshClick}
              className="border-red-300 hover:bg-red-100 dark:border-red-700 dark:hover:bg-red-900"
            >
              Start Fresh Chat
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
