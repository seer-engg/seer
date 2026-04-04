import { Pencil, Trash2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formatMemoryDate, truncateContent, type Memory } from "@/lib/memory-api";
import { cn } from "@/lib/utils";

interface MemoryRowProps {
  memory: Memory;
  onEdit: () => void;
  onDelete: () => void;
  isMutating: boolean;
}

export function MemoryRow({ memory, onEdit, onDelete, isMutating }: MemoryRowProps) {
  const displayContent = truncateContent(memory.memory);

  return (
    <div
      className={cn(
        "flex items-start gap-3 py-3 px-3 rounded-lg transition-colors",
        "hover:bg-muted/50"
      )}
    >
      <div className="flex-1 min-w-0">
        <p className="text-sm line-clamp-2">{displayContent}</p>
        <p className="text-xs text-muted-foreground mt-1">
          {formatMemoryDate(memory.created_at)}
        </p>
      </div>
      <div className="flex items-center gap-1 flex-shrink-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={onEdit}
          disabled={isMutating}
          className="h-7 w-7 p-0 text-muted-foreground"
          title="Edit memory"
          aria-label={`Edit memory: ${displayContent}`}
        >
          <Pencil className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDelete}
          disabled={isMutating}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
          title="Delete memory"
          aria-label={`Delete memory: ${displayContent}`}
        >
          {isMutating ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
    </div>
  );
}
