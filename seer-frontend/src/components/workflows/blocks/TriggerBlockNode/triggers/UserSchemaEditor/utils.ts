/**
 * Utilities for converting between SchemaField[] and JSON Schema.
 */

import type { JsonObject } from '@/types/workflow-spec';
import type { SchemaField, SchemaFieldType } from './types';

let fieldIdCounter = 0;

/**
 * Generates a unique ID for a schema field.
 */
export function generateFieldId(): string {
  fieldIdCounter += 1;
  return `field_${Date.now()}_${fieldIdCounter}`;
}

/**
 * Converts a simple field list to a JSON Schema object.
 */
export function fieldsToJsonSchema(fields: SchemaField[]): JsonObject {
  const properties: JsonObject = {};
  const required: string[] = [];

  for (const field of fields) {
    if (!field.name.trim()) {
      continue;
    }

    const prop: JsonObject = {
      type: field.type,
    };

    if (field.description.trim()) {
      prop.description = field.description.trim();
    }

    properties[field.name.trim()] = prop;

    if (field.required) {
      required.push(field.name.trim());
    }
  }

  const schema: JsonObject = {
    type: 'object',
    properties,
    additionalProperties: false,
  };

  if (required.length > 0) {
    schema.required = required;
  }

  return schema;
}

/**
 * Maps JSON Schema type to SchemaFieldType.
 */
function mapJsonSchemaType(schemaType: unknown): SchemaFieldType {
  if (typeof schemaType === 'string') {
    switch (schemaType) {
      case 'string':
        return 'string';
      case 'integer':
        return 'integer';
      case 'number':
        return 'number';
      case 'boolean':
        return 'boolean';
      case 'object':
        return 'object';
      case 'array':
        return 'array';
      default:
        return 'string';
    }
  }

  // Handle array of types (e.g., ["string", "null"])
  if (Array.isArray(schemaType)) {
    const nonNullType = schemaType.find((t) => t !== 'null');
    if (nonNullType) {
      return mapJsonSchemaType(nonNullType);
    }
  }

  return 'string';
}

/**
 * Converts a JSON Schema to a simple field list.
 * Best effort - complex schemas may lose some information.
 */
export function jsonSchemaToFields(schema: JsonObject): SchemaField[] {
  const fields: SchemaField[] = [];

  const properties = schema.properties as JsonObject | undefined;
  if (!properties || typeof properties !== 'object') {
    return fields;
  }

  const requiredFields = Array.isArray(schema.required)
    ? (schema.required as string[])
    : [];

  for (const [name, propValue] of Object.entries(properties)) {
    if (typeof propValue !== 'object' || propValue === null) {
      continue;
    }

    const propSchema = propValue as JsonObject;

    fields.push({
      id: generateFieldId(),
      name,
      type: mapJsonSchemaType(propSchema.type),
      description: typeof propSchema.description === 'string' ? propSchema.description : '',
      required: requiredFields.includes(name),
    });
  }

  return fields;
}

/**
 * Validates a JSON Schema string and returns the parsed schema or an error.
 */
export function validateJsonSchema(
  schemaString: string,
): { valid: true; schema: JsonObject } | { valid: false; error: string } {
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

  const schema = parsed as JsonObject;

  // Basic JSON Schema validation
  if (schema.type !== 'object') {
    return { valid: false, error: 'Schema type must be "object"' };
  }

  if (schema.properties && typeof schema.properties !== 'object') {
    return { valid: false, error: 'Schema properties must be an object' };
  }

  return { valid: true, schema };
}

/**
 * Validates a field name.
 */
export function validateFieldName(name: string): { valid: boolean; error?: string } {
  const trimmed = name.trim();

  if (!trimmed) {
    return { valid: false, error: 'Field name is required' };
  }

  // No character restrictions - backend supports spaces, hyphens, special chars, Unicode
  return { valid: true };
}

/**
 * Creates a default empty schema.
 */
export function createEmptySchema(): JsonObject {
  return {
    type: 'object',
    properties: {},
    additionalProperties: false,
  };
}

/**
 * Creates a default field.
 */
export function createDefaultField(): SchemaField {
  return {
    id: generateFieldId(),
    name: '',
    type: 'string',
    description: '',
    required: false,
  };
}
