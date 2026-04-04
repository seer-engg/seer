/**
 * Graph Layout Utility
 *
 * Uses dagre to compute hierarchical tree layout positions for workflow nodes.
 * Dagre implements the Sugiyama algorithm for layered graph drawing.
 */
import dagre from 'dagre';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';

export interface LayoutOptions {
  /** Layout direction: TB = top-bottom, LR = left-right */
  direction?: 'TB' | 'LR';
  /** Width of each node (default: 240px based on BaseBlockNode maxWidth) */
  nodeWidth?: number;
  /** Height of each node (default: 80px) */
  nodeHeight?: number;
  /** Horizontal spacing between nodes at the same rank */
  nodesep?: number;
  /** Vertical spacing between ranks/levels */
  ranksep?: number;
  /** Horizontal margin around the graph */
  marginx?: number;
  /** Vertical margin around the graph */
  marginy?: number;
}

const DEFAULT_OPTIONS: Required<LayoutOptions> = {
  direction: 'TB',
  nodeWidth: 240,
  nodeHeight: 80,
  nodesep: 50,
  ranksep: 60,
  marginx: 50,
  marginy: 50,
};

/**
 * Computes layouted positions for all nodes using dagre's hierarchical layout algorithm.
 *
 * The function:
 * 1. Separates trigger nodes from workflow nodes (triggers go above the graph)
 * 2. Creates a dagre graph with the workflow nodes and edges
 * 3. Runs the layout algorithm
 * 4. Converts dagre's center-based coordinates to React Flow's top-left origin
 *
 * @param nodes - Array of workflow nodes
 * @param edges - Array of workflow edges
 * @param options - Layout configuration options
 * @returns Object with layouted nodes and original edges
 */
export function getLayoutedElements(
  nodes: Node<WorkflowNodeData>[],
  edges: WorkflowEdge[],
  options?: LayoutOptions
): { nodes: Node<WorkflowNodeData>[]; edges: WorkflowEdge[] } {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Handle empty graph
  if (nodes.length === 0) {
    return { nodes: [], edges };
  }

  // Separate trigger nodes from workflow nodes
  // Triggers are positioned separately above the main workflow
  const triggerNodes = nodes.filter((n) => n.data?.type === 'trigger');
  const workflowNodes = nodes.filter((n) => n.data?.type !== 'trigger');

  // If no workflow nodes, just position triggers
  if (workflowNodes.length === 0) {
    const layoutedTriggers = triggerNodes.map((node, index) => ({
      ...node,
      position: {
        x: index * (opts.nodeWidth + opts.nodesep),
        y: 0,
      },
    }));
    return { nodes: layoutedTriggers, edges };
  }

  // Create dagre graph
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: opts.direction,
    nodesep: opts.nodesep,
    ranksep: opts.ranksep,
    marginx: opts.marginx,
    marginy: opts.marginy,
  });
  g.setDefaultEdgeLabel(() => ({}));

  // Add workflow nodes to the graph
  workflowNodes.forEach((node) => {
    g.setNode(node.id, {
      width: opts.nodeWidth,
      height: opts.nodeHeight,
    });
  });

  // Add edges (only between workflow nodes, skip trigger edges)
  const triggerNodeIds = new Set(triggerNodes.map((t) => t.id));
  edges
    .filter((e) => !triggerNodeIds.has(e.source))
    .forEach((edge) => {
      // Only add edge if both source and target are in the workflow nodes
      if (g.hasNode(edge.source) && g.hasNode(edge.target)) {
        g.setEdge(edge.source, edge.target);
      }
    });

  // Run the layout algorithm
  dagre.layout(g);

  // Apply computed positions to workflow nodes
  // Dagre returns center coordinates, React Flow uses top-left origin
  const layoutedWorkflowNodes = workflowNodes.map((node) => {
    const nodeWithPosition = g.node(node.id);
    return {
      ...node,
      position: {
        x: nodeWithPosition.x - opts.nodeWidth / 2,
        y: nodeWithPosition.y - opts.nodeHeight / 2,
      },
    };
  });

  // Find the topmost position of workflow nodes to position triggers above
  const minWorkflowY = Math.min(...layoutedWorkflowNodes.map((n) => n.position.y));

  // Position trigger nodes in a row above the main workflow graph
  const triggerRowY = minWorkflowY - opts.nodeHeight - opts.ranksep;

  // Center triggers horizontally relative to the workflow
  const workflowXs = layoutedWorkflowNodes.map((n) => n.position.x);
  const workflowCenterX =
    workflowXs.length > 0
      ? (Math.min(...workflowXs) + Math.max(...workflowXs) + opts.nodeWidth) / 2
      : 0;

  const totalTriggerWidth =
    triggerNodes.length * opts.nodeWidth + (triggerNodes.length - 1) * opts.nodesep;
  const triggerStartX = workflowCenterX - totalTriggerWidth / 2;

  const layoutedTriggerNodes = triggerNodes.map((node, index) => ({
    ...node,
    position: {
      x: triggerStartX + index * (opts.nodeWidth + opts.nodesep),
      y: triggerRowY,
    },
  }));

  return {
    nodes: [...layoutedTriggerNodes, ...layoutedWorkflowNodes],
    edges, // Edges don't need position updates - React Flow handles them
  };
}
