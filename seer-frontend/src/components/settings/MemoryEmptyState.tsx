import { Brain, Search } from "lucide-react";
import { Button } from "@/components/ui/button";

interface MemoryEmptyStateProps {
  hasSearch: boolean;
  onAdd?: () => void;
}

export function MemoryEmptyState({ hasSearch, onAdd }: MemoryEmptyStateProps) {
  return (
    <div className="py-8 text-center">
      <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-3">
        {hasSearch ? (
          <Search className="h-6 w-6 text-muted-foreground/50" />
        ) : (
          <Brain className="h-6 w-6 text-muted-foreground/50" />
        )}
      </div>
      <p className="text-sm text-muted-foreground">
        {hasSearch ? "No memories match your search" : "No memories yet"}
      </p>
      <p className="text-xs text-muted-foreground mt-1">
        {hasSearch
          ? "Try adjusting your search query"
          : "Memories are created during AI conversations"}
      </p>
      {!hasSearch && onAdd && (
        <Button variant="outline" size="sm" className="mt-4" onClick={onAdd}>
          Add your first memory
        </Button>
      )}
    </div>
  );
}
