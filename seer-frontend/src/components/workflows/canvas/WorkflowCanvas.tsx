/**
 * Enhanced Workflow Canvas Component
 *
 * Supports drag-and-drop blocks, custom node types, and connection validation.
 * Based on ReactFlow (@xyflow/react).
 */
import { useCallback, useMemo, useRef } from 'react';
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  Node,
  ConnectionMode,
  type NodeTypes,
  type EdgeTypes,
  type DefaultEdgeOptions,
  type OnInit,
  MarkerType,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { cn } from '@/lib/utils';
import { WorkflowCanvasContext } from './workflow-canvas-context';
import { FloatingActions } from '@/components/general/FloatingActions';
import { WorkflowNodeData, WorkflowEdge, DroppedBlockData } from '../types';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import { TriggerBlockNode } from '../blocks/TriggerBlockNode';

// Import custom node types
import { ToolBlockNode } from '../blocks/ToolBlockNode';
import { IfElseBlockNode } from '../blocks/IfElseBlockNode';
import { ForLoopBlockNode } from '../blocks/ForLoopBlockNode';
import { MCPBlockNode } from '../blocks/MCPBlockNode';
import { HITLBlockNode } from '../blocks/HITLBlockNode';
import { EABotBlockNode } from '../blocks/EABotBlockNode';
import { BrowserBlockNode } from '../blocks/BrowserBlockNode';
import { ImageGenBlockNode } from '../blocks/ImageGenBlockNode';
import { AgentBlockNode } from '../blocks/AgentBlockNode';
import { CustomEdge } from './CustomEdge';
import { Plus } from 'lucide-react';
import { useCanvasStore } from '@/stores';
import { useUIStore } from '@/stores/uiStore';
import { useCanvasDragDrop } from '../../../hooks/useCanvasDragDrop';
import { useConnectionValidation } from '../../../hooks/useConnectionValidation';
import { useRenderedNodes } from './hooks/useRenderedNodes';
import { useWorkflowChanges } from './hooks/useWorkflowChanges';
import { useNodeHandlers } from './hooks/useNodeHandlers';
import { InlineBlockPicker } from './InlineBlockPicker';

function EmptyCanvasPlaceholder() {
  const setInlineBlockPicker = useUIStore((state) => state.setInlineBlockPicker);
  const rightPaneMode = useUIStore((state) => state.rightPaneMode);
  const isPaneOpen = rightPaneMode !== null;

  const handleClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    setInlineBlockPicker({
      visible: true,
      sourceNodeId: '',
      position: { x: e.clientX, y: e.clientY },
    });
  };

  return (
    <div
      className="absolute inset-0 flex items-center justify-center pointer-events-none transition-[padding-right] duration-200 ease-in-out"
      style={{ paddingRight: isPaneOpen ? 360 : 0 }}
    >
      <div className="flex flex-col items-center gap-4 pointer-events-auto select-none">
        <button
          onClick={handleClick}
          className="flex items-center justify-center w-14 h-14 rounded-full border-2 border-dashed border-border bg-card hover:border-primary hover:bg-primary/5 transition-colors cursor-pointer"
          aria-label="Add first block"
        >
          <Plus className="w-6 h-6 text-muted-foreground" />
        </button>
        <div className="text-center">
          <p className="text-sm font-medium text-foreground">Start building</p>
          <p className="text-xs text-muted-foreground mt-0.5">Click to add your first block or trigger</p>
        </div>
      </div>
    </div>
  );
}

/**
 * Extract tool names from workflow nodes
 */
// eslint-disable-next-line react-refresh/only-export-components
export function getToolNamesFromNodes(nodes: Node<WorkflowNodeData>[]): string[] {
  return nodes
    .filter((node) => node.data.type === 'tool')
    .map((node) => {
      const config = node.data.config as ToolBlockConfig | undefined;
      return config?.tool_name || config?.toolName || '';
    })
    .filter(Boolean);
}

const nodeTypes: NodeTypes = {
  tool: ToolBlockNode,
  if_else: IfElseBlockNode,
  for_loop: ForLoopBlockNode,
  mcp: MCPBlockNode,
  hitl: HITLBlockNode,
  browser: BrowserBlockNode,
  image_gen: ImageGenBlockNode,
  agent: AgentBlockNode,
  ea_bot: EABotBlockNode,
  trigger: TriggerBlockNode,
};

const edgeTypes: EdgeTypes = {
  default: CustomEdge,
};

const DEFAULT_EDGE_OPTIONS: DefaultEdgeOptions = {
  style: { strokeWidth: 3 },
  markerEnd: { type: MarkerType.ArrowClosed },
};

// Stable references — prevents React Flow from re-registering internal listeners on every render
const handleReactFlowInit: OnInit = (instance) => instance.fitView({ padding: 0.2 });
const PAN_ON_DRAG = [0, 1, 2];
const DELETE_KEY_CODE = ['Backspace', 'Delete'];

interface WorkflowCanvasProps {
  previewGraph?: { nodes?: Node<WorkflowNodeData>[]; edges?: WorkflowEdge[] } | null;
  onNodeClick?: (node: Node<WorkflowNodeData>) => void;
  onNodeDrop?: (
    blockData: DroppedBlockData,
    position: { x: number; y: number },
  ) => void;
  className?: string;
  readOnly?: boolean;
}

export function WorkflowCanvas({
  previewGraph = null,
  onNodeClick,
  onNodeDrop,
  className,
  readOnly = false,
}: WorkflowCanvasProps) {
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const setNodes = useCanvasStore((state) => state.setNodes);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const setSelectedNodeId = useCanvasStore((state) => state.setSelectedNodeId);
  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const updateNode = useCanvasStore((state) => state.updateNode);
  const closeRightPane = useUIStore((state) => state.closeRightPane);
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const workflowNodes = previewGraph?.nodes ?? nodes;
  const workflowEdges = previewGraph?.edges ?? edges;

  const renderedNodes = useRenderedNodes(workflowNodes, selectedNodeId);

  const updateNodeData = useCallback(
    (nodeId: string, updates: Partial<WorkflowNodeData>) => updateNode(nodeId, updates),
    [updateNode],
  );

  const { handleNodesChange, handleEdgesChange } = useWorkflowChanges(readOnly, selectedNodeId);
  const { onConnect } = useConnectionValidation({ readOnly, workflowNodes, workflowEdges, setEdges });
  const { handleNodeClick } = useNodeHandlers(readOnly, setSelectedNodeId, onNodeClick);
  const { onDragOver, onDrop } = useCanvasDragDrop({ readOnly, setNodes, onNodeDrop });

  const handlePaneClick = useCallback(() => {
    if (readOnly) return;
    closeRightPane();
    setSelectedNodeId(null);
  }, [readOnly, closeRightPane, setSelectedNodeId]);

  const contextValue = useMemo(
    () => ({
      nodes: workflowNodes,
      edges: workflowEdges,
      updateNodeData,
      readOnly,
    }),
    [workflowNodes, workflowEdges, updateNodeData, readOnly],
  );

  return (
    <WorkflowCanvasContext.Provider value={contextValue}>
      <div
        ref={reactFlowWrapper}
        data-testid="workflow-canvas"
        className={cn('relative w-full h-full bg-[hsl(var(--canvas-bg))]', className)}
        onDrop={onDrop}
        onDragOver={onDragOver}
      >
        <ReactFlow
          nodes={renderedNodes}
          edges={workflowEdges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={readOnly ? undefined : onConnect}
          onNodeClick={handleNodeClick}
          onPaneClick={handlePaneClick}
          connectionMode={ConnectionMode.Strict}
          onInit={handleReactFlowInit}
          nodesDraggable={!readOnly}
          nodesConnectable={!readOnly}
          elementsSelectable={!readOnly}
          panOnDrag={PAN_ON_DRAG}
          panOnScroll
          zoomOnScroll={false}
          zoomOnPinch
          minZoom={0.1}
          maxZoom={2}
          selectNodesOnDrag={!readOnly}
          fitView={readOnly}
          deleteKeyCode={readOnly ? null : DELETE_KEY_CODE}
          defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
        >
          <Background
            variant={BackgroundVariant.Dots}
            gap={20}
            size={1}
            color="hsl(var(--canvas-grid))"
          />
          {!readOnly && (
            <Controls
              showInteractive={false}
              className="!bg-card !border-border !rounded-lg !shadow-lg"
            />
          )}
        </ReactFlow>

        {/* Floating Actions for Settings & Theme */}
        {!readOnly && <FloatingActions />}

        {/* Empty canvas placeholder - shown when no nodes exist */}
        {workflowNodes.length === 0 && !readOnly && <EmptyCanvasPlaceholder />}

        {/* Inline block picker - portal-rendered near "+" button click position */}
        {!readOnly && <InlineBlockPicker />}
      </div>
    </WorkflowCanvasContext.Provider>
  );
}
