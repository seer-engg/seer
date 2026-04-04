import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import type { BlockSelectionPayload } from '@/types/block-selection';
import { generateNodeId, withDefaultBlockConfig } from '@/lib/workflow-nodes';

/**
 * Calculate the midpoint position between two nodes
 */
export function calculateMidpoint(
  sourceNode: Node<WorkflowNodeData>,
  targetNode: Node<WorkflowNodeData>
): { x: number; y: number } {
  const midX = (sourceNode.position.x + targetNode.position.x) / 2;
  const midY = (sourceNode.position.y + targetNode.position.y) / 2;

  // TODO: Future enhancement - check for collisions with existing nodes
  // and adjust position if needed

  return { x: midX, y: midY };
}

/**
 * Create an edge from source to target with optional branch type
 */
export function createEdgeFromSource(
  sourceNodeId: string,
  targetNodeId: string,
  branch?: 'true' | 'false' | 'loop' | 'exit'
): WorkflowEdge {
  const edgeId = `edge-${sourceNodeId}-${targetNodeId}${branch ? `-${branch}` : ''}`;

  const edge: WorkflowEdge = {
    id: edgeId,
    source: sourceNodeId,
    target: targetNodeId,
    type: 'default',
  };

  // Add branch data if provided (for control flow nodes)
  if (branch) {
    edge.data = { branch };
    edge.sourceHandle = branch; // Control flow nodes have named handles
  }

  return edge;
}

/**
 * Insert a new node between an existing edge
 * Returns updated nodes and edges arrays
 */
export function insertNodeBetweenEdge(
  edge: WorkflowEdge,
  newNodeConfig: BlockSelectionPayload,
  nodes: Node<WorkflowNodeData>[],
  edges: WorkflowEdge[],
  functionBlocksMap: Map<string, unknown>
): { nodes: Node<WorkflowNodeData>[]; edges: WorkflowEdge[] } {
  // Find source and target nodes
  const sourceNode = nodes.find((n) => n.id === edge.source);
  const targetNode = nodes.find((n) => n.id === edge.target);

  if (!sourceNode || !targetNode) {
    throw new Error('Source or target node not found');
  }

  // Calculate midpoint position for new node
  const position = calculateMidpoint(sourceNode, targetNode);

  // Create new node with default config
  const defaultConfig = withDefaultBlockConfig(
    newNodeConfig.type,
    newNodeConfig.config,
    functionBlocksMap
  );

  const existingNodeIds = nodes.map((n) => n.id);

  // Build context for descriptive ID generation
  const context: Parameters<typeof generateNodeId>[2] = {
    blockType: newNodeConfig.type,
  };

  // Add type-specific context
  if (newNodeConfig.type === 'tool' && newNodeConfig.config?.tool_name) {
    context.toolName = newNodeConfig.config.tool_name as string;
  } else if (newNodeConfig.type === 'llm' && defaultConfig.model) {
    context.model = defaultConfig.model as string;
  }

  const nodeId = generateNodeId('block', existingNodeIds, context);

  const newNode: Node<WorkflowNodeData> = {
    id: nodeId,
    type: newNodeConfig.type as WorkflowNodeData['type'],
    position,
    data: {
      type: newNodeConfig.type as WorkflowNodeData['type'],
      label: nodeId, // Use generated ID as label
      config: defaultConfig,
    },
  };

  // Create two new edges: source → new node → target
  const firstEdge = createEdgeFromSource(
    edge.source,
    newNode.id,
    edge.data?.branch // Preserve branch type from original edge
  );

  const secondEdge = createEdgeFromSource(newNode.id, edge.target);

  // Remove old edge and add new node + edges
  const updatedEdges = edges.filter((e) => e.id !== edge.id).concat([firstEdge, secondEdge]);
  const updatedNodes = [...nodes, newNode];

  return { nodes: updatedNodes, edges: updatedEdges };
}
