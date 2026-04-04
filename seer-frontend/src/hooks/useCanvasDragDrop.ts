import { useCallback } from 'react';
import { useReactFlow, type Node } from '@xyflow/react';
import type { WorkflowNodeData, DroppedBlockData } from '../components/workflows/types';

type SetNodes = (
  nodes:
    | Node<WorkflowNodeData>[]
    | ((nodes: Node<WorkflowNodeData>[]) => Node<WorkflowNodeData>[]),
) => void;

type DropPosition = { x: number; y: number };

interface DroppedTool {
  name: string;
  slug?: string;
  provider: string;
  integration_type: string;
  output_schema?: unknown;
}

interface DroppedToolGroup {
  integration_type: string;
  defaultTool: DroppedTool;
}

interface BlockDragData {
  blockType: string;
  label: string;
}

interface ToolDragData {
  tool: DroppedTool;
}

interface TriggerDragData {
  triggerKey: string;
  title?: string;
}

type DragData =
  | (BlockDragData & { type: 'block' })
  | (ToolDragData & { type: 'tool' })
  | (TriggerDragData & { type: 'trigger' })
  | ( { integrationGroup: DroppedToolGroup } & { type: 'tool-group' });

const createNodeFromPayload = (payload: DroppedBlockData, position: DropPosition): Node<WorkflowNodeData> => ({
  id: `node-${Date.now()}`,
  type: payload.type,
  position,
  data: {
    type: payload.type,
    label: payload.label,
    config: payload.config ?? {},
  },
});

const addNodeOrCallback = (
  payload: DroppedBlockData,
  position: DropPosition,
  onNodeDrop: UseCanvasDragDropParams['onNodeDrop'],
  setNodes: SetNodes,
) => {
  if (onNodeDrop) {
    onNodeDrop(payload, position);
    return;
  }
  const newNode = createNodeFromPayload(payload, position);
  setNodes((nds) => [...nds, newNode]);
};

const buildToolConfig = (tool: DroppedTool, integrationType?: string) => ({
  tool_name: tool.slug || tool.name,
  provider: tool.provider,
  integration_type: integrationType ?? tool.integration_type,
  ...(tool.output_schema ? { output_schema: tool.output_schema } : {}),
  params: {},
});

const handleBlockDrop = (
  dragData: BlockDragData,
  addNode: (payload: DroppedBlockData) => void,
) => {
  addNode({
    type: dragData.blockType,
    label: dragData.label,
    config: {},
  });
};

const handleToolGroupDrop = (
  dragData: { integrationGroup: DroppedToolGroup },
  addNode: (payload: DroppedBlockData) => void,
) => {
  const { integrationGroup } = dragData;
  const { defaultTool } = integrationGroup;
  if (!defaultTool) return;

  addNode({
    type: 'tool',
    label: defaultTool.name,
    config: buildToolConfig(defaultTool, integrationGroup.integration_type),
  });
};

const handleToolDrop = (
  dragData: ToolDragData,
  addNode: (payload: DroppedBlockData) => void,
) => {
  const tool = dragData.tool;
  if (!tool) return;

  addNode({
    type: 'tool',
    label: tool.name,
    config: buildToolConfig(tool),
  });
};

const handleTriggerDrop = (
  dragData: TriggerDragData,
  position: DropPosition,
  onNodeDrop?: UseCanvasDragDropParams['onNodeDrop'],
) => {
  if (!onNodeDrop) {
    console.log('[WorkflowCanvas] Trigger dropped without onNodeDrop handler, ignoring');
    return;
  }
  onNodeDrop(
    {
      type: 'trigger',
      label: dragData.title || dragData.triggerKey,
      config: {
        triggerKey: dragData.triggerKey,
      },
    },
    position,
  );
};

interface UseCanvasDragDropParams {
  readOnly?: boolean;
  setNodes: SetNodes;
  onNodeDrop?: (
    blockData: DroppedBlockData,
    position: { x: number; y: number },
  ) => void;
}

export function useCanvasDragDrop({ readOnly = false, setNodes, onNodeDrop }: UseCanvasDragDropParams) {
  const { screenToFlowPosition } = useReactFlow();

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      if (readOnly) return;

      const data = event.dataTransfer.getData('application/reactflow');
      if (!data) return;

      try {
        const dragData = JSON.parse(data) as DragData;
        const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
        const addNode = (payload: DroppedBlockData) => addNodeOrCallback(payload, position, onNodeDrop, setNodes);

        const dropHandlers: Record<DragData['type'], () => void> = {
          block: () => handleBlockDrop(dragData, addNode),
          'tool-group': () => handleToolGroupDrop(dragData, addNode),
          tool: () => handleToolDrop(dragData, addNode),
          trigger: () => handleTriggerDrop(dragData, position, onNodeDrop),
        };

        const handler = dropHandlers[dragData.type as keyof typeof dropHandlers];
        handler?.();
      } catch (error) {
        console.error('Error processing drop:', error);
      }
    },
    [readOnly, screenToFlowPosition, setNodes, onNodeDrop],
  );

  return { onDragOver, onDrop } as const;
}
