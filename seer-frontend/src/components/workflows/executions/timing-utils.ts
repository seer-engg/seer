import type { WorkflowNodeTrace, RunHistoryEntry } from './types';

export interface NodeWithTiming extends WorkflowNodeTrace {
  startOffset: number; // seconds from run start
  duration: number; // duration in seconds
}

/**
 * Get display name for a node, combining label from execution graph with node type
 */
export function getNodeDisplayName(
  nodeId: string,
  nodeType: string,
  executionGraph?: RunHistoryEntry['execution_graph']
): string {
  const node = executionGraph?.nodes.find((n) => n.id === nodeId);
  const label = node?.label || nodeId.slice(0, 8);
  return `${label} (${nodeType})`;
}

/**
 * Calculate relative start time and duration for each node
 */
export function calculateNodeTiming(
  nodes: WorkflowNodeTrace[],
  runStartTime?: string | null
): NodeWithTiming[] {
  if (!runStartTime || nodes.length === 0) {
    return nodes.map((node) => ({
      ...node,
      startOffset: 0,
      duration: 1,
    }));
  }

  const startMs = new Date(runStartTime).getTime();
  const firstNode = nodes[0];

  // Calculate initialization gap before first node
  const firstNodeStartMs = firstNode?.timestamp
    ? new Date(firstNode.timestamp).getTime()
    : startMs;
  const initDuration = Math.max(0, (firstNodeStartMs - startMs) / 1000);

  // Calculate timing for real nodes
  const realNodesWithTiming: NodeWithTiming[] = nodes.map((node, index) => {
    const nodeStartMs = node.timestamp ? new Date(node.timestamp).getTime() : startMs;
    const startOffset = Math.max(0, (nodeStartMs - startMs) / 1000);

    // Duration: time until next node (or estimate 1s if last node)
    const nextNode = nodes[index + 1];
    const duration = nextNode?.timestamp
      ? Math.max(0.1, (new Date(nextNode.timestamp).getTime() - nodeStartMs) / 1000)
      : 1;

    return {
      ...node,
      startOffset,
      duration,
    };
  });

  // Insert synthetic trigger node if gap > 0.5s
  if (initDuration > 0.5) {
    const triggerNode: NodeWithTiming = {
      node_id: '__trigger_init__',
      node_type: 'trigger',
      is_synthetic: true,
      startOffset: 0,
      duration: initDuration,
      timestamp: runStartTime,
    };
    return [triggerNode, ...realNodesWithTiming];
  }

  return realNodesWithTiming;
}

/**
 * Calculate total duration of the run
 */
export function calculateTotalDuration(
  startTime?: string | null,
  endTime?: string | null
): number {
  if (!startTime || !endTime) return 0;
  return Math.max(0, (new Date(endTime).getTime() - new Date(startTime).getTime()) / 1000);
}

/**
 * Format duration as human-readable string
 */
export function formatDuration(seconds: number): string {
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

/**
 * Generate time markers for the timeline axis
 */
export function generateTimeMarkers(totalDuration: number, count: number = 5): number[] {
  if (totalDuration === 0) return [0];
  const interval = totalDuration / (count - 1);
  return Array.from({ length: count }, (_, i) => i * interval);
}

/**
 * Get color classes for node type
 */
export function getNodeTypeColors(nodeType: string): string {
  const type = nodeType.toLowerCase();
  if (type.includes('tool')) {
    return 'bg-amber-500/20 border-amber-500';
  }
  if (type.includes('llm') || type.includes('agent')) {
    return 'bg-purple-500/20 border-purple-500';
  }
  if (type.includes('if') || type.includes('condition')) {
    return 'bg-blue-500/20 border-blue-500';
  }
  if (type.includes('loop') || type.includes('for')) {
    return 'bg-emerald-500/20 border-emerald-500';
  }
  // Default
  return 'bg-muted/20 border-muted';
}

/**
 * Get border color for node status
 */
export function getNodeStatusBorderColor(hasError: boolean): string {
  return hasError ? 'border-bug' : 'border-emerald-500';
}
