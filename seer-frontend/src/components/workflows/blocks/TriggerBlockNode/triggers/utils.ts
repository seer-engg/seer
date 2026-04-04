import type { JsonObject } from '@/types/workflow-spec';

/**
 * Infer schema for primitive values (string, number, boolean, null).
 */
function inferPrimitiveSchema(value: unknown, key: string): JsonObject | null {
  if (value === null) {
    return { type: 'string' } as JsonObject;
  }
  if (typeof value === 'string') {
    return { type: 'string', title: key } as JsonObject;
  }
  if (typeof value === 'number') {
    return {
      type: Number.isInteger(value) ? 'integer' : 'number',
      title: key,
    } as JsonObject;
  }
  if (typeof value === 'boolean') {
    return { type: 'boolean', title: key } as JsonObject;
  }
  return null;
}

/**
 * Infer schema for items in an array based on the first element.
 */
function inferArrayItemSchema(firstItem: unknown): JsonObject {
  const primitiveSchema = inferPrimitiveSchema(firstItem, '');
  if (primitiveSchema) {
    return primitiveSchema;
  }

  if (Array.isArray(firstItem)) {
    return { type: 'array', items: { type: 'string' } } as JsonObject;
  }

  if (typeof firstItem === 'object') {
    return inferSchemaFromPayload(firstItem as Record<string, unknown>) as JsonObject;
  }

  return { type: 'string' } as JsonObject;
}

/**
 * Infer schema for a single property value.
 */
function inferPropertySchema(key: string, value: unknown): JsonObject {
  const primitiveSchema = inferPrimitiveSchema(value, key);
  if (primitiveSchema) {
    return primitiveSchema;
  }

  if (Array.isArray(value)) {
    const itemsSchema = value.length > 0
      ? inferArrayItemSchema(value[0])
      : ({ type: 'string' } as JsonObject);
    return { type: 'array', title: key, items: itemsSchema } as JsonObject;
  }

  if (typeof value === 'object') {
    const schema = inferSchemaFromPayload(value as Record<string, unknown>) as JsonObject;
    (schema as Record<string, unknown>).title = key;
    return schema;
  }

  return { type: 'string', title: key } as JsonObject;
}

/**
 * Infer a JSON Schema from a plain JS object (runtime payload).
 */
export function inferSchemaFromPayload(payload: Record<string, unknown>): JsonObject {
  const properties: Record<string, JsonObject> = {};
  const required: string[] = [];

  for (const [key, value] of Object.entries(payload)) {
    required.push(key);
    properties[key] = inferPropertySchema(key, value);
  }

  return {
    type: 'object',
    properties,
    required,
  } as JsonObject;
}
