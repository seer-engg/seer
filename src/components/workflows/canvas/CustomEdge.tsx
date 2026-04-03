import { useCallback } from 'react';
import { EdgeProps, getSmoothStepPath, useReactFlow, MarkerType } from '@xyflow/react';
import type { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';
import { EdgeButtonGroup } from './EdgeButtonGroup';
import { useCanvasStore } from '@/stores';
import type { Node } from '@xyflow/react';

/**
 * Custom edge component with hover interactions
 * Shows delete button at edge midpoint on hover
 * Supports trigger edge styling (purple dashed) and branch labels
 */
export function CustomEdge({
  id, source, sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: EdgeProps<WorkflowEdge>) {
  const { getNode } = useReactFlow();
  const deleteEdge = useCanvasStore((state) => state.deleteEdge);

  // Determine if this is a trigger edge (entry point, styled differently)
  const sourceNode = getNode(source) as Node<WorkflowNodeData> | undefined;
  const isTriggerEdge = sourceNode?.type === 'trigger';

  // Calculate smooth step path for edge rendering with rounded corners
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 8,  // Rounded corners on 90° turns (matches node card radius)
    offset: 20,       // Minimum distance before first turn
  });

  // Handle delete edge button click
  const handleDelete = useCallback(() => {
    deleteEdge(id);
    useCanvasStore.getState().markDirty();
  }, [id, deleteEdge]);

  // Determine edge styling (trigger edges have purple dashed style)
  const edgeStyle = isTriggerEdge
    ? {
        stroke: 'hsl(239 84% 67%)',
        strokeWidth: 3,
        strokeDasharray: '5,5',
      }
    : {
        strokeWidth: 3,
        ...style,
      };

  return (
    <g className="react-flow__edge custom-edge group">
      {/* Invisible wider path for easier hover detection */}
      <path
        d={edgePath}
        stroke="transparent"
        strokeWidth={20}
        fill="none"
        className="react-flow__edge-interaction"
      />

      {/* Visible edge path */}
      <path
        id={id}
        d={edgePath}
        style={edgeStyle}
        fill="none"
        className="react-flow__edge-path"
        markerEnd={markerEnd || `url(#${MarkerType.ArrowClosed})`}
      />

      {/* Branch label for control flow edges */}
      {data?.branch && (
        <text
          x={labelX}
          y={labelY - 10}
          className="text-xs fill-muted-foreground pointer-events-none"
          textAnchor="middle"
        >
          {data.branch}
        </text>
      )}

      {/* Delete button at midpoint (visible on hover) */}
      <foreignObject
        x={labelX - 16}
        y={labelY - 12}
        width={32}
        height={24}
        className="overflow-visible opacity-0 group-hover:opacity-100 transition-opacity duration-200"
        style={{ pointerEvents: 'all' }}
      >
        <EdgeButtonGroup onDelete={handleDelete} />
      </foreignObject>
    </g>
  );
}
