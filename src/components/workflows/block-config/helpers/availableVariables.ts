import { Node } from '@xyflow/react';

import { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';
import type { InputDef, TriggerSpec } from '@/types/workflow-spec';
import { useVariablesStore } from '@/stores/variablesStore';


interface NodeOutputMetadata {
  identifier: string;
  properties: string[];
}

const INPUT_IDENTIFIER = 'inputs';

export const collectAvailableVariables = (
  allNodes: Node<WorkflowNodeData>[] = [],
  allEdges: WorkflowEdge[] = [],
  currentNode?: Node<WorkflowNodeData> | null,
  workflowInputs?: Record<string, InputDef>,
  triggers?: TriggerSpec[],
): string[] => {
  if (!currentNode) {
    return [];
  }

  const nodeMap = new Map(allNodes.map((node) => [node.id, node]));
  const ancestorIds = findAncestorNodeIds(currentNode.id, allEdges);
  const suggestions = new Set<string>();

  ancestorIds.forEach((nodeId) => {
    const node = nodeMap.get(nodeId);
    if (!node) {
      return;
    }
    const metadata = buildNodeOutputMetadata(node);
    addMetadataToSuggestions(metadata, suggestions);
  });

  addWorkflowInputsToSuggestions(workflowInputs, suggestions);
  addTriggersToSuggestions(triggers, suggestions);
  addGlobalVariablesToSuggestions(suggestions);

  return Array.from(suggestions).sort();
};

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

function buildNodeOutputMetadata(node: Node<WorkflowNodeData>): NodeOutputMetadata | null {
  const nodeType = node.data?.type;
  if (!nodeType) {
    return null;
  }
  if (nodeType === 'trigger') {
    return null;
  }

  const identifier = getOutputIdentifier(node);
  if (!identifier) {
    return null;
  }

  // Special handling for browser nodes - they always have wrapper fields
  if (nodeType === 'browser') {
    return buildBrowserNodeOutputMetadata(node, identifier);
  }

  // Special handling for ea_bot nodes - fixed output structure
  if (nodeType === 'ea_bot') {
    return {
      identifier,
      properties: [
        'bot_id',
        'state',
        'transcript',
        'transcript[0]',
        'transcript[0].speaker',
        'transcript[0].text',
        'transcript[0].start_ms',
        'transcript[0].duration_ms',
        'recording_url',
      ],
    };
  }

  const schema = node.data?.config?.output_schema;
  const properties = extractSchemaProperties(schema);

  return {
    identifier,
    properties,
  };
}

/**
 * Build output metadata for browser nodes.
 * Browser nodes always output a wrapper structure with standard fields:
 * - success: boolean
 * - result: string
 * - extracted_data: custom schema properties
 * - final_url: string | null
 * - screenshots: array of file objects
 */
function buildBrowserNodeOutputMetadata(
  node: Node<WorkflowNodeData>,
  identifier: string
): NodeOutputMetadata {
  const properties: string[] = [
    // Standard browser output fields (always available)
    'success',
    'result',
    'final_url',
    'screenshots',
    'screenshots[0]',
    'screenshots[0].file_id',
    'screenshots[0].filename',
    'screenshots[0].url',
  ];

  // Add extracted_data and its nested properties from custom schema
  const customSchema = node.data?.config?.output_schema;
  if (customSchema && typeof customSchema === 'object' && 'properties' in customSchema) {
    properties.push('extracted_data');
    const schemaProps = customSchema.properties as Record<string, unknown>;
    Object.keys(schemaProps).forEach((prop) => {
      properties.push(`extracted_data.${prop}`);
    });
  }

  return { identifier, properties };
}

function getOutputIdentifier(node: Node<WorkflowNodeData>): string | null {
  // Node ID is the variable reference identifier
  return node.id || null;
}

function extractSchemaProperties(schema: unknown): string[] {
  return Array.from(collectSchemaPaths(schema)).sort();
}

function collectSchemaPaths(schema: unknown, prefix = ''): Set<string> {
  const paths = new Set<string>();

  if (!isSchemaRecord(schema)) {
    return paths;
  }

  const schemaType = schema.type;

  if (schemaType === 'object' && isSchemaRecord(schema.properties)) {
    Object.entries(schema.properties).forEach(([key, value]) => {
      const nextPath = prefix ? `${prefix}.${key}` : key;
      paths.add(nextPath);
      collectSchemaPaths(value, nextPath).forEach((childPath) => paths.add(childPath));
    });
  } else if (schemaType === 'array' && 'items' in schema) {
    const itemsSchema = Array.isArray(schema.items) ? schema.items[0] : schema.items;
    const indexPath = prefix ? `${prefix}[0]` : '[0]';
    paths.add(indexPath);
    collectSchemaPaths(itemsSchema, indexPath).forEach((childPath) => paths.add(childPath));
  }

  return paths;
}

function isSchemaRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function addMetadataToSuggestions(
  metadata: NodeOutputMetadata | null,
  suggestions: Set<string>,
): void {
  if (!metadata || !metadata.identifier) {
    return;
  }

  suggestions.add(metadata.identifier);
  metadata.properties.forEach((property) => {
    suggestions.add(`${metadata.identifier}.${property}`);
  });
}

function addWorkflowInputsToSuggestions(
  workflowInputs: Record<string, InputDef> | undefined,
  suggestions: Set<string>,
) {
  if (!workflowInputs || Object.keys(workflowInputs).length === 0) {
    return;
  }
  const names = Object.keys(workflowInputs).sort();
  addMetadataToSuggestions(
    {
      identifier: INPUT_IDENTIFIER,
      properties: names,
    },
    suggestions,
  );
}

function addTriggersToSuggestions(
  triggers: TriggerSpec[] | undefined,
  suggestions: Set<string>,
): void {
  if (!triggers || triggers.length === 0) {
    return;
  }

  triggers.forEach((trigger) => {
    // Use trigger ID directly - no sanitization
    const identifier = trigger.id;

    if (!identifier) {
      return;
    }

    const eventSchema = trigger.event_schema;
    const properties = extractSchemaProperties(eventSchema);

    addMetadataToSuggestions(
      {
        identifier,
        properties,
      },
      suggestions,
    );
  });
}

function addGlobalVariablesToSuggestions(suggestions: Set<string>): void {
  const variables = useVariablesStore.getState().variables;
  if (!variables || variables.length === 0) {
    return;
  }
  const keys = variables.map((v) => v.key);
  addMetadataToSuggestions({ identifier: 'vars', properties: keys }, suggestions);
}
