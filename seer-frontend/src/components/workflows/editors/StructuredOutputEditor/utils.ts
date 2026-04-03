/**
 * Utilities for converting between FieldDefinition[] and JSON Schema.
 */

import type { FieldDefinition, JsonSchema, JsonSchemaNode } from './types';

let fieldIdCounter = 0;

export function generateFieldId(): string {
  fieldIdCounter += 1;
  return `field_${Date.now()}_${fieldIdCounter}`;
}

export const PYDANTIC_TYPES = [
  { value: 'any', label: 'Any' },
  { value: 'str', label: 'String' },
  { value: 'int', label: 'Integer' },
  { value: 'float', label: 'Float' },
  { value: 'bool', label: 'Boolean' },
  { value: 'list', label: 'List' },
  { value: 'dict', label: 'Dictionary' },
];

export function createEmptyField(): FieldDefinition {
  return {
    id: generateFieldId(),
    name: '',
    type: 'str',
    description: '',
  };
}

/** Pydantic type → JSON Schema type */
function pydanticToJsonType(pydanticType: string): string {
  const map: Record<string, string> = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
    list: 'array',
    dict: 'object',
    any: 'string',
  };
  return map[pydanticType] ?? 'string';
}

/** JSON Schema type → Pydantic type */
function jsonToPydanticType(jsonType: string): string {
  const map: Record<string, string> = {
    string: 'str',
    integer: 'int',
    number: 'float',
    boolean: 'bool',
    array: 'list',
    object: 'dict',
  };
  return map[jsonType] ?? 'any';
}

/** Recursively convert a properties map to FieldDefinition[] */
function propsToFields(
  properties: Record<string, JsonSchemaNode>,
): FieldDefinition[] {
  return Object.entries(properties).map(([name, node]) => {
    const pydanticType = jsonToPydanticType(node.type);
    const field: FieldDefinition = {
      id: generateFieldId(),
      name,
      type: pydanticType,
      description: node.description ?? '',
    };

    if (pydanticType === 'dict' && node.properties) {
      field.children = propsToFields(node.properties);
    }

    if (pydanticType === 'list' && node.items) {
      if (node.items.type === 'object' && node.items.properties) {
        field.itemType = 'object';
        field.itemChildren = propsToFields(node.items.properties);
      } else {
        field.itemType = jsonToPydanticType(node.items.type);
      }
    }

    return field;
  });
}

/** Convert JSON Schema to FieldDefinition[] */
export function schemaToFields(
  schema: JsonSchema | Record<string, unknown> | undefined,
): FieldDefinition[] {
  if (!schema || !('properties' in schema) || !schema.properties) {
    return [];
  }
  return propsToFields(
    schema.properties as Record<string, JsonSchemaNode>,
  );
}

function withDesc(node: JsonSchemaNode, desc: string | undefined): JsonSchemaNode {
  const trimmed = desc?.trim();
  return trimmed ? { ...node, description: trimmed } : node;
}

function buildListItems(field: FieldDefinition): JsonSchemaNode | undefined {
  if (field.itemType === 'object') {
    const itemNode: JsonSchemaNode = { type: 'object' };
    if (field.itemChildren?.length) {
      itemNode.properties = fieldsToProps(field.itemChildren);
      const required = Object.keys(itemNode.properties);
      if (required.length > 0) itemNode.required = required;
    }
    return itemNode;
  }
  return field.itemType ? { type: pydanticToJsonType(field.itemType) } : undefined;
}

function fieldToNode(field: FieldDefinition): JsonSchemaNode {
  if (field.type === 'dict') {
    const node = withDesc({ type: 'object' }, field.description);
    if (field.children?.length) {
      node.properties = fieldsToProps(field.children);
      const required = Object.keys(node.properties);
      if (required.length > 0) node.required = required;
    }
    return node;
  }
  if (field.type === 'list') {
    const node = withDesc({ type: 'array' }, field.description);
    const items = buildListItems(field);
    if (items) node.items = items;
    return node;
  }
  return withDesc({ type: pydanticToJsonType(field.type) }, field.description);
}

/** Recursively convert FieldDefinition[] to a JSON Schema properties map */
function fieldsToProps(
  fields: FieldDefinition[],
): Record<string, JsonSchemaNode> {
  const properties: Record<string, JsonSchemaNode> = {};

  for (const field of fields) {
    const trimmedName = field.name.trim();
    if (!trimmedName) continue;
    properties[trimmedName] = fieldToNode(field);
  }

  return properties;
}

/** Convert FieldDefinition[] to a full JSON Schema */
export function fieldsToSchema(fields: FieldDefinition[]): JsonSchema {
  const properties = fieldsToProps(fields);
  const required = Object.keys(properties);
  return {
    type: 'object',
    properties,
    ...(required.length > 0 ? { required } : {}),
  };
}

/** Validates a raw JSON string as a valid output schema */
export function validateJsonSchema(
  schemaString: string,
): { valid: true; schema: JsonSchema } | { valid: false; error: string } {
  if (!schemaString.trim()) {
    return { valid: false, error: 'Schema cannot be empty' };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(schemaString);
  } catch (e) {
    return { valid: false, error: `Invalid JSON: ${(e as Error).message}` };
  }

  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { valid: false, error: 'Schema must be a JSON object' };
  }

  const schema = parsed as Record<string, unknown>;

  if (schema['type'] !== 'object') {
    return { valid: false, error: 'Schema type must be "object"' };
  }

  if (schema['properties'] !== undefined && typeof schema['properties'] !== 'object') {
    return { valid: false, error: 'Schema properties must be an object' };
  }

  return {
    valid: true,
    schema: parsed as JsonSchema,
  };
}
