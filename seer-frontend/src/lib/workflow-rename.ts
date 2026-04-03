import type { Node } from '@xyflow/react';

import type { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';

/** Valid node ID format: alphanumeric + underscore, must start with a letter */
const NODE_ID_REGEX = /^[a-zA-Z][a-zA-Z0-9_]*$/;

/** Maximum length for node IDs */
const MAX_NODE_ID_LENGTH = 64;

/** Reserved IDs that cannot be used (they have special meaning in the workflow system) */
const RESERVED_IDS = new Set(['inputs', 'outputs', 'item', 'index', 'loop']);

export interface NodeIdValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Validates a proposed node ID for format, uniqueness, and reserved words.
 *
 * @param proposedId - The new ID to validate
 * @param existingNodeIds - Array of all current node IDs in the workflow
 * @param currentNodeId - The node's current ID (excluded from uniqueness check)
 * @returns Validation result with error message if invalid
 */
export function validateNodeId(
  proposedId: string,
  existingNodeIds: string[],
  currentNodeId: string,
): NodeIdValidationResult {
  const trimmedId = proposedId.trim();

  // Check non-empty
  if (!trimmedId) {
    return { valid: false, error: 'Node ID cannot be empty' };
  }

  // Check max length
  if (trimmedId.length > MAX_NODE_ID_LENGTH) {
    return { valid: false, error: `Node ID cannot exceed ${MAX_NODE_ID_LENGTH} characters` };
  }

  // Check format (alphanumeric + underscore, starts with letter)
  if (!NODE_ID_REGEX.test(trimmedId)) {
    return {
      valid: false,
      error: 'Node ID must start with a letter and contain only letters, numbers, and underscores',
    };
  }

  // Check reserved words
  if (RESERVED_IDS.has(trimmedId.toLowerCase())) {
    return { valid: false, error: `"${trimmedId}" is a reserved word` };
  }

  // Check uniqueness (excluding current node)
  const otherIds = existingNodeIds.filter((id) => id !== currentNodeId);
  if (otherIds.includes(trimmedId)) {
    return { valid: false, error: 'A node with this ID already exists' };
  }

  return { valid: true };
}

/**
 * Replaces node ID references within a template string.
 * Handles both mustache {{nodeId.prop}} and dollar ${nodeId.prop} formats.
 *
 * @param text - The template string to process
 * @param oldId - The old node ID to find
 * @param newId - The new node ID to replace with
 * @returns The template string with updated references
 */
export function replaceNodeIdInTemplate(text: string, oldId: string, newId: string): string {
  if (!text || typeof text !== 'string') {
    return text;
  }

  // Escape special regex characters in the oldId
  const escapedOldId = oldId.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  // Replace mustache format: {{oldId.property}} or {{oldId}}
  // The pattern matches the nodeId followed by either a dot+property or end of expression
  const mustachePattern = new RegExp(
    `(\\{\\{\\s*)${escapedOldId}((?:\\.[^}]*)?\\s*\\}\\})`,
    'g',
  );
  let result = text.replace(mustachePattern, `$1${newId}$2`);

  // Replace dollar format: ${oldId.property} or ${oldId}
  const dollarPattern = new RegExp(
    `(\\$\\{\\s*)${escapedOldId}((?:\\.[^}]*)?\\s*\\})`,
    'g',
  );
  result = result.replace(dollarPattern, `$1${newId}$2`);

  return result;
}

/**
 * Recursively traverses an object/array and replaces node ID references in all string values.
 * Returns a new object (immutable operation).
 *
 * @param value - The value to process (can be any type)
 * @param oldId - The old node ID to find
 * @param newId - The new node ID to replace with
 * @returns A new value with updated references
 */
export function replaceReferencesDeep<T>(value: T, oldId: string, newId: string): T {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === 'string') {
    return replaceNodeIdInTemplate(value, oldId, newId) as T;
  }

  if (Array.isArray(value)) {
    return value.map((item) => replaceReferencesDeep(item, oldId, newId)) as T;
  }

  if (typeof value === 'object') {
    const result: Record<string, unknown> = {};
    for (const [key, val] of Object.entries(value)) {
      result[key] = replaceReferencesDeep(val, oldId, newId);
    }
    return result as T;
  }

  // Primitives (numbers, booleans) pass through unchanged
  return value;
}

/**
 * Generates a consistent edge ID from source, target, and optional branch.
 */
function generateEdgeId(source: string, target: string, branch?: string): string {
  return branch ? `${source}-${target}-${branch}` : `${source}-${target}`;
}

export interface RenameNodeResult {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
}

/**
 * Renames a node in the workflow graph, updating all references atomically.
 *
 * This function:
 * 1. Updates the target node's `id` and `data.label`
 * 2. Updates all edges where `source === oldId` or `target === oldId`
 * 3. Regenerates edge IDs to reflect new node ID
 * 4. Updates template references in ALL other nodes' `config` objects
 *
 * @param oldId - The current node ID
 * @param newId - The new node ID to use
 * @param nodes - Current workflow nodes
 * @param edges - Current workflow edges
 * @returns New nodes and edges arrays with updated references
 */
export function renameNodeInGraph(
  oldId: string,
  newId: string,
  nodes: Node<WorkflowNodeData>[],
  edges: WorkflowEdge[],
): RenameNodeResult {
  // Update nodes
  const updatedNodes = nodes.map((node) => {
    if (node.id === oldId) {
      // This is the node being renamed - update its ID and label
      return {
        ...node,
        id: newId,
        data: {
          ...node.data,
          label: newId,
        },
      };
    }

    // For all other nodes, update any template references in their config
    if (node.data?.config) {
      const updatedConfig = replaceReferencesDeep(node.data.config, oldId, newId);
      if (updatedConfig !== node.data.config) {
        return {
          ...node,
          data: {
            ...node.data,
            config: updatedConfig,
          },
        };
      }
    }

    return node;
  });

  // Update edges
  const updatedEdges = edges.map((edge) => {
    const newSource = edge.source === oldId ? newId : edge.source;
    const newTarget = edge.target === oldId ? newId : edge.target;

    // If neither source nor target changed, return original edge
    if (newSource === edge.source && newTarget === edge.target) {
      return edge;
    }

    // Regenerate edge ID with new node ID
    const branch = edge.data?.branch;
    const newEdgeId = generateEdgeId(newSource, newTarget, branch);

    return {
      ...edge,
      id: newEdgeId,
      source: newSource,
      target: newTarget,
    };
  });

  return { nodes: updatedNodes, edges: updatedEdges };
}
