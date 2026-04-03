import { LayoutGrid } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAutoLayout } from "@/hooks/useAutoLayout";

export function FloatingActions() {
  const { organizeGraph, canOrganize } = useAutoLayout();

  return (
    <div
      className="absolute bottom-6 right-6 flex flex-col gap-3 z-40"
      data-floating-actions
    >
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="default"
            size="icon"
            onClick={organizeGraph}
            disabled={!canOrganize}
            aria-label="Organize layout"
            className="h-12 w-12 rounded-full shadow-lg hover:shadow-xl transition-all hover:scale-105 hover:border-seer/50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <LayoutGrid className="h-5 w-5" />
          </Button>
        </TooltipTrigger>
        <TooltipContent side="left">
          <p>Organize Layout</p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
}
