import { Node } from '@xyflow/react';

import { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';

/**
 * Represents a file-like output from an upstream node
 */
export interface FileVariable {
  /** Full path to reference the file (e.g., "download-1.file") */
  path: string;
  /** Source node ID */
  nodeId: string;
  /** Display name for the node */
  nodeName: string;
  /** Inferred file type based on property name patterns */
  inferredType: 'image' | 'pdf' | 'document' | 'text' | 'unknown';
}

/**
 * Patterns that indicate a property is likely a file output
 */
const FILE_PROPERTY_PATTERNS = [
  'file',
  'attachment',
  'download',
  'content',
  'data',
  'blob',
  'binary',
];

/**
 * Patterns that suggest specific file types
 */
const TYPE_INFERENCE_PATTERNS: Record<string, FileVariable['inferredType']> = {
  image: 'image',
  photo: 'image',
  picture: 'image',
  screenshot: 'image',
  img: 'image',
  thumbnail: 'image',
  pdf: 'pdf',
  doc: 'document',
  document: 'document',
  word: 'document',
  docx: 'document',
  text: 'text',
  txt: 'text',
  content: 'text',
};

/**
 * Infer file type from property name
 */
function inferFileType(propertyName: string): FileVariable['inferredType'] {
  const nameLower = propertyName.toLowerCase();

  for (const [pattern, fileType] of Object.entries(TYPE_INFERENCE_PATTERNS)) {
    if (nameLower.includes(pattern)) {
      return fileType;
    }
  }

  return 'unknown';
}

/**
 * Check if a property name looks like it could be a file output
 */
function isLikelyFileProperty(propertyName: string): boolean {
  const nameLower = propertyName.toLowerCase();
  return FILE_PROPERTY_PATTERNS.some((pattern) => nameLower.includes(pattern));
}

/**
 * Find all ancestor node IDs by traversing edges backwards
 * (Reused from availableVariables.ts pattern)
 */
function findAncestorNodeIds(nodeId: string, edges: WorkflowEdge[] = []): string[] {
  const incomingMap = edges.reduce<Map<string, WorkflowEdge[]>>((map, edge) => {
    if (!edge?.target) {
      return map;
    }
    const existing = map.get(edge.target) ?? [];
    existing.push(edge);
    map.set(edge.target, existing);
    return map;
  }, new Map());

  const visited = new Set<string>();
  const stack = [nodeId];

  while (stack.length > 0) {
    const current = stack.pop();
    if (!current) {
      continue;
    }
    const incomingEdges = incomingMap.get(current) ?? [];
    incomingEdges.forEach((edge) => {
      const sourceId = edge.source;
      if (!sourceId || sourceId === nodeId || visited.has(sourceId)) {
        return;
      }
      visited.add(sourceId);
      stack.push(sourceId);
    });
  }

  return Array.from(visited);
}

/**
 * Extract file-like properties from a JSON schema
 */
function extractFileProperties(
  schema: unknown,
  prefix = '',
): Array<{ path: string; inferredType: FileVariable['inferredType'] }> {
  const results: Array<{ path: string; inferredType: FileVariable['inferredType'] }> = [];

  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) {
    return results;
  }

  const schemaObj = schema as Record<string, unknown>;
  const schemaType = schemaObj.type;

  if (schemaType === 'object' && schemaObj.properties && typeof schemaObj.properties === 'object') {
    const properties = schemaObj.properties as Record<string, unknown>;

    for (const [key, value] of Object.entries(properties)) {
      const propertyPath = prefix ? `${prefix}.${key}` : key;

      // Check if this property looks like a file
      if (isLikelyFileProperty(key)) {
        results.push({
          path: propertyPath,
          inferredType: inferFileType(key),
        });
      }

      // Recurse into nested objects
      const nested = extractFileProperties(value, propertyPath);
      results.push(...nested);
    }
  } else if (schemaType === 'array' && schemaObj.items) {
    // Check array items schema
    const itemsSchema = Array.isArray(schemaObj.items) ? schemaObj.items[0] : schemaObj.items;
    const indexPath = prefix ? `${prefix}[0]` : '[0]';
    const nested = extractFileProperties(itemsSchema, indexPath);
    results.push(...nested);
  }

  return results;
}

/**
 * Get display name for a node
 */
function getNodeDisplayName(node: Node<WorkflowNodeData>): string {
  return node.data?.label || node.id;
}

/**
 * Check if a tool name suggests file output capability
 */
function isFileProducingTool(toolName: string): boolean {
  const toolNameLower = toolName.toLowerCase();
  return (
    toolNameLower.includes('download') ||
    toolNameLower.includes('attachment') ||
    toolNameLower.includes('export') ||
    toolNameLower.includes('fetch')
  );
}

/**
 * Extract file variables from a single node's schema
 */
function extractNodeFileVariables(
  node: Node<WorkflowNodeData>,
  nodeId: string,
): FileVariable[] {
  const results: FileVariable[] = [];
  const nodeName = getNodeDisplayName(node);
  const outputSchema = node.data?.config?.output_schema;

  if (outputSchema) {
    const fileProps = extractFileProperties(outputSchema);
    for (const { path, inferredType } of fileProps) {
      results.push({
        path: `${nodeId}.${path}`,
        nodeId,
        nodeName,
        inferredType,
      });
    }
  }

  return results;
}

/**
 * Add generic file reference for tool nodes that likely output files
 */
function addToolFileReference(
  node: Node<WorkflowNodeData>,
  nodeId: string,
  existingVariables: FileVariable[],
): FileVariable | null {
  const toolName = (node.data?.config?.tool_name || node.data?.config?.toolName || '') as string;

  if (!isFileProducingTool(toolName)) {
    return null;
  }

  const existingPaths = existingVariables.filter((v) => v.nodeId === nodeId).map((v) => v.path);
  const genericPath = `${nodeId}.file`;

  if (existingPaths.includes(genericPath)) {
    return null;
  }

  return {
    path: genericPath,
    nodeId,
    nodeName: getNodeDisplayName(node),
    inferredType: inferFileType(toolName),
  };
}

/**
 * Discover file-like variables from upstream nodes
 */
export function discoverFileVariables(
  allNodes: Node<WorkflowNodeData>[] = [],
  allEdges: WorkflowEdge[] = [],
  currentNode: Node<WorkflowNodeData> | null,
): FileVariable[] {
  if (!currentNode) {
    return [];
  }

  const nodeMap = new Map(allNodes.map((node) => [node.id, node]));
  const ancestorIds = findAncestorNodeIds(currentNode.id, allEdges);
  const fileVariables: FileVariable[] = [];

  for (const nodeId of ancestorIds) {
    const node = nodeMap.get(nodeId);
    if (!node || node.data?.type === 'trigger') {
      continue;
    }

    // Extract from schema
    fileVariables.push(...extractNodeFileVariables(node, nodeId));

    // Add generic file ref for tool nodes
    if (node.data?.type === 'tool') {
      const toolRef = addToolFileReference(node, nodeId, fileVariables);
      if (toolRef) {
        fileVariables.push(toolRef);
      }
    }

    // Add file reference for image_gen nodes (backend outputs 'file' property)
    if (node.data?.type === 'image_gen') {
      fileVariables.push({
        path: `${nodeId}.file`,
        nodeId,
        nodeName: getNodeDisplayName(node),
        inferredType: 'image',
      });
    }
  }

  return fileVariables.sort((a, b) =>
    a.nodeId !== b.nodeId ? a.nodeId.localeCompare(b.nodeId) : a.path.localeCompare(b.path)
  );
}
