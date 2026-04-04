import { Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface EdgeButtonGroupProps {
  onDelete: () => void;
}

/**
 * Button group displayed at edge midpoint on hover
 * Shows delete button for quick edge deletion
 */
export function EdgeButtonGroup({ onDelete }: EdgeButtonGroupProps) {
  return (
    <div className="flex bg-card border border-border rounded-md shadow-lg p-1 edge-button-group">
      <Button
        size="icon"
        variant="ghost"
        data-testid="edge-delete-button"
        className="h-6 w-6 hover:bg-destructive hover:text-destructive-foreground"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        aria-label="Delete edge"
        title="Delete connection"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  );
}
