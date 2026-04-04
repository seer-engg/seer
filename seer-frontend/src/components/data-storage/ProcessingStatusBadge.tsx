import { Clock, Loader2, CheckCircle2, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { DocumentProcessingStatus } from "@/lib/knowledge-api";

interface ProcessingStatusBadgeProps {
  status: DocumentProcessingStatus;
  error?: string;
  className?: string;
}

const statusConfig: Record<
  DocumentProcessingStatus,
  {
    label: string;
    icon: typeof Clock;
    className: string;
    animate?: boolean;
  }
> = {
  pending: {
    label: "Pending",
    icon: Clock,
    className: "bg-muted text-muted-foreground",
  },
  processing: {
    label: "Processing",
    icon: Loader2,
    className: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
    animate: true,
  },
  completed: {
    label: "Completed",
    icon: CheckCircle2,
    className: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
  },
  failed: {
    label: "Failed",
    icon: AlertTriangle,
    className: "bg-destructive/10 text-destructive",
  },
};

export function ProcessingStatusBadge({
  status,
  error,
  className,
}: ProcessingStatusBadgeProps) {
  const config = statusConfig[status];
  const Icon = config.icon;

  const badge = (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium",
        config.className,
        className
      )}
    >
      <Icon
        className={cn("h-3 w-3", config.animate && "animate-spin")}
      />
      <span>{config.label}</span>
    </div>
  );

  // Wrap failed status with tooltip to show error message
  if (status === "failed" && error) {
    return (
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>{badge}</TooltipTrigger>
          <TooltipContent side="top" className="max-w-xs">
            <p className="text-sm">{error}</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    );
  }

  return badge;
}

export type { ProcessingStatusBadgeProps };
