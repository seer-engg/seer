/**
 * Node Delete Button Component
 *
 * Small "X" button that appears on hover over workflow nodes.
 * Uses React Flow's deleteElements to trigger the standard deletion flow,
 * ensuring trigger cleanup and edge removal happen automatically.
 */
import { Trash2 } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';

interface NodeDeleteButtonProps {
  nodeId: string;
}

export function NodeDeleteButton({ nodeId }: NodeDeleteButtonProps) {
  const { deleteElements } = useReactFlow();

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    deleteElements({ nodes: [{ id: nodeId }] });
  };

  return (
    <button
      className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-destructive/80 hover:bg-destructive text-destructive-foreground flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 shadow-sm"
      onClick={handleDelete}
      onPointerDown={(e) => e.stopPropagation()}
      aria-label="Delete node"
      title="Delete node"
    >
      <Trash2 className="h-3 w-3" />
    </button>
  );
}
