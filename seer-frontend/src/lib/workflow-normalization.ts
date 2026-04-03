import type { Node, Edge } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge, FunctionBlockSchema } from '@/components/workflows/types';
import type { RawEdge } from '@/types/workflow-api';
import { withDefaultBlockConfig } from './workflow-nodes';

const isBranchValue = (value: unknown): value is 'true' | 'false' =>
  value === 'true' || value === 'false';

function extractBranchValue(rawEdge: RawEdge, edgeData?: Record<string, unknown>): 'true' | 'false' | undefined {
  const legacyBranch = edgeData?.branch ?? (rawEdge as Record<string, unknown>)?.branch;
  const legacyHandle = rawEdge.targetHandle || edgeData?.targetHandle;
  const branchCandidate = legacyBranch ?? legacyHandle;
  return isBranchValue(branchCandidate) ? branchCandidate : undefined;
}

function normalizeEdgeData(
  edgeData: Record<string, unknown> | undefined,
  branch: 'true' | 'false' | undefined,
): Record<string, unknown> | undefined {
  let normalizedData = edgeData ? { ...edgeData } : undefined;
  if (branch) {
    normalizedData = { ...(normalizedData || {}), branch };
  }
  if (normalizedData && Object.keys(normalizedData).length === 0) {
    return undefined;
  }
  return normalizedData;
}

/**
 * Normalizes a raw edge object to WorkflowEdge format
 *
 * IMPORTANT: This function preserves sourceHandle and targetHandle properties.
 * These are required by React Flow to determine which Handle component an edge
 * visually connects from/to. For conditional/loop edges, sourceHandle must match
 * the branch value ('true'/'false'/'loop'/'exit') to render from the correct handle.
 */
export function normalizeEdge(edge: unknown): WorkflowEdge {
  if (typeof edge !== 'object' || edge === null) {
    throw new Error('Invalid edge: expected object');
  }

  const rawEdge = edge as RawEdge;
  const edgeData = (rawEdge as Record<string, unknown>)?.data as Record<string, unknown> | undefined;
  const branch = extractBranchValue(rawEdge, edgeData);
  const normalizedData = normalizeEdgeData(edgeData, branch);

  return {
    ...rawEdge,
    data: normalizedData,
  } as WorkflowEdge;
}

/**
 * Normalizes an array of raw edges
 */
export function normalizeEdges(rawEdges?: unknown[]): WorkflowEdge[] {
  if (!Array.isArray(rawEdges)) {
    return [];
  }

  return rawEdges.map(normalizeEdge);
}

function shouldIncludeNode(node: unknown): boolean {
  const rawNode = node as Record<string, unknown>;
  const dataType = (rawNode?.data as Record<string, unknown>)?.type;
  const nodeType = dataType ?? rawNode?.type;
  return nodeType !== 'input';
}

function extractNodeType(rawNode: Record<string, unknown>, data: Record<string, unknown>): WorkflowNodeData['type'] {
  return (rawNode?.type || data?.type || 'tool') as WorkflowNodeData['type'];
}

function extractNodeData(rawNode: Record<string, unknown>): Record<string, unknown> {
  return (rawNode?.data ?? {}) as Record<string, unknown>;
}

function extractNodePosition(rawNode: Record<string, unknown>): { x: number; y: number } {
  return (rawNode?.position ?? { x: 0, y: 0 }) as { x: number; y: number };
}

function buildNodeData(
  data: Record<string, unknown>,
  resolvedType: WorkflowNodeData['type'],
  rawNode: Record<string, unknown>,
  configWithDefaults: Record<string, unknown>,
): WorkflowNodeData {
  return {
    ...data,
    type: data?.type || resolvedType,
    label: data?.label ?? rawNode?.id ?? '',
    config: configWithDefaults,
  } as WorkflowNodeData;
}

function transformNode(
  node: unknown,
  functionBlockMap?: Map<string, FunctionBlockSchema>,
): Node<WorkflowNodeData> {
  const rawNode = node as Record<string, unknown>;
  const data = extractNodeData(rawNode);
  const position = extractNodePosition(rawNode);
  const resolvedType = extractNodeType(rawNode, data);
  const configWithDefaults = withDefaultBlockConfig(
    resolvedType,
    (data?.config ?? {}) as Record<string, unknown>,
    functionBlockMap,
  );
  const nodeData = buildNodeData(data, resolvedType, rawNode, configWithDefaults);

  return {
    ...rawNode,
    type: resolvedType,
    position,
    data: nodeData,
  } as Node<WorkflowNodeData>;
}

/**
 * Normalizes an array of raw nodes to React Flow format
 */
export function normalizeNodes(
  rawNodes?: unknown[],
  functionBlockMap?: Map<string, FunctionBlockSchema>,
): Node<WorkflowNodeData>[] {
  if (!Array.isArray(rawNodes)) {
    return [];
  }

  return rawNodes
    .filter(shouldIncludeNode)
    .map((node) => transformNode(node, functionBlockMap));
}
