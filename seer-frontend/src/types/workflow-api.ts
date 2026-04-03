/**
 * Raw API types for workflow data (before normalization)
 */

/**
 * Raw node from API response (before React Flow normalization)
 */
export interface RawNode {
  id: string;
  type: string;
  position?: { x: number; y: number };
  data?: unknown;
  [key: string]: unknown;
}

/**
 * Raw edge from API response (before React Flow normalization)
 */
export interface RawEdge {
  id?: string;
  source: string;
  target: string;
  sourceHandle?: string | null;
  targetHandle?: string | null;
  [key: string]: unknown;
}

/**
 * Type guards for runtime validation
 */

export function isRawNode(value: unknown): value is RawNode {
  if (typeof value !== 'object' || value === null) return false;
  const node = value as Record<string, unknown>;
  return (
    typeof node.id === 'string' &&
    typeof node.type === 'string'
  );
}

export function isRawEdge(value: unknown): value is RawEdge {
  if (typeof value !== 'object' || value === null) return false;
  const edge = value as Record<string, unknown>;
  return (
    typeof edge.source === 'string' &&
    typeof edge.target === 'string'
  );
}
