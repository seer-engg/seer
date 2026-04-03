import type { CSSProperties } from 'react';
import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useUIStore } from '@/stores/uiStore';
import type { CanvasNodeType } from '../types';
import { getDefaultOutputButtonStyle, useHandleOrientation } from './hooks/useHandleOrientation';

interface OutputEdgeButtonProps {
  nodeId: string;
  nodeType: CanvasNodeType;
  branch?: 'true' | 'false' | 'loop' | 'exit';
  position?: CSSProperties;
  readOnly?: boolean;
}

/**
 * Small floating "+" button positioned on edge extending from output handle
 * Always visible with subtle scale effect on hover
 * Clicking shows the inline block picker near the button position
 */
export function OutputEdgeButton({
  nodeId,
  nodeType,
  branch,
  position,
  readOnly = false,
}: OutputEdgeButtonProps) {
  const setPendingConnection = useUIStore((state) => state.setPendingConnection);
  const setInlineBlockPicker = useUIStore((state) => state.setInlineBlockPicker);
  const orientation = useHandleOrientation();

  if (readOnly) return null;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent node selection

    setPendingConnection({
      mode: 'append',
      sourceNodeId: nodeId,
      targetNodeId: null,
      edgeId: null,
      branch,
    });

    // Show inline block picker near the click position instead of switching right panel tab
    setInlineBlockPicker({
      visible: true,
      sourceNodeId: nodeId,
      position: { x: e.clientX, y: e.clientY },
    });
  };

  const defaultStyle = getDefaultOutputButtonStyle(orientation);
  const mergedStyle: CSSProperties = {
    ...defaultStyle,
    ...position,
    transform: position?.transform ?? defaultStyle.transform,
  };

  return (
    <Button
      size="icon"
      variant="ghost"
      className="output-edge-button absolute h-5 w-5 rounded-full bg-seer/80 hover:bg-seer opacity-100 scale-100 hover:scale-110 transition-all duration-200"
      style={{
        ...mergedStyle,
      }}
      onClick={handleClick}
      aria-label={`Add node after ${nodeType}${branch ? ` (${branch})` : ''}`}
      title={`Add block after this node${branch ? ` (${branch} branch)` : ''}`}
    >
      <Plus className="h-3 w-3" />
    </Button>
  );
}
