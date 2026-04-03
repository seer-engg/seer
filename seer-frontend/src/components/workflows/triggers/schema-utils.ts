/**
 * Utilities for extracting workflow inputs from JSON Schema definitions
 * and deriving quick options from trigger event schemas.
 */

import type { InputDef, JsonObject } from '@/types/workflow-spec';

export interface QuickOption {
  label: string;
  path: string;
}

type JsonSchemaType = string | string[] | undefined;

/**
 * Maps JSON Schema type to InputDef type.
 * Handles union types like ["string", "null"] by picking the first non-null type.
 */
export function mapJsonSchemaTypeToInputDefType(
  schemaType: JsonSchemaType,
): InputDef['type'] {
  if (!schemaType) {
    return 'string';
  }

  // Handle array of types (e.g., ["string", "null"])
  if (Array.isArray(schemaType)) {
    const nonNullType = schemaType.find((t) => t !== 'null');
    if (nonNullType) {
      return mapJsonSchemaTypeToInputDefType(nonNullType);
    }
    return 'string';
  }

  // Map JSON Schema types to InputDef types
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

/**
 * Extracts the 'data' properties from an event schema.
 * Event schemas typically have structure: { properties: { data: { properties: {...} } } }
 */
function getDataPropertiesFromEventSchema(
  eventSchema: JsonObject | null | undefined,
): JsonObject | null {
  if (!eventSchema) {
    return null;
  }

  const properties = eventSchema.properties as JsonObject | undefined;
  if (!properties) {
    return null;
  }

  const dataSchema = properties.data as JsonObject | undefined;
  if (!dataSchema) {
    return null;
  }

  // Check if data has explicit properties defined
  const dataProperties = dataSchema.properties as JsonObject | undefined;
  if (dataProperties && typeof dataProperties === 'object') {
    return {
      properties: dataProperties,
      required: dataSchema.required,
      additionalProperties: dataSchema.additionalProperties,
    };
  }

  // If data has additionalProperties: true (like webhook/form), return null
  // to indicate user needs to define their own schema
  if (dataSchema.additionalProperties === true) {
    return null;
  }

  return null;
}

/**
 * Extracts InputDef records from event_schema.properties.data.properties.
 * Returns an empty record if the schema has flexible data (additionalProperties: true).
 */
export function extractInputsFromEventSchema(
  eventSchema: JsonObject | null | undefined,
): Record<string, InputDef> {
  const dataSchema = getDataPropertiesFromEventSchema(eventSchema);
  if (!dataSchema) {
    return {};
  }

  const properties = dataSchema.properties as JsonObject;
  const requiredFields = Array.isArray(dataSchema.required)
    ? (dataSchema.required as string[])
    : [];

  const inputs: Record<string, InputDef> = {};

  for (const [key, value] of Object.entries(properties)) {
    if (typeof value !== 'object' || value === null) {
      continue;
    }

    const propSchema = value as JsonObject;
    const schemaType = propSchema.type as JsonSchemaType;

    inputs[key] = {
      type: mapJsonSchemaTypeToInputDefType(schemaType),
      description: (propSchema.description as string) || undefined,
      required: requiredFields.includes(key),
    };
  }

  return inputs;
}

/**
 * Checks if an event schema has flexible data (additionalProperties: true).
 * This is used for webhook/form triggers where users define their own schema.
 */
export function hasFlexibleDataSchema(
  eventSchema: JsonObject | null | undefined,
): boolean {
  if (!eventSchema) {
    return false;
  }

  const properties = eventSchema.properties as JsonObject | undefined;
  if (!properties) {
    return false;
  }

  const dataSchema = properties.data as JsonObject | undefined;
  if (!dataSchema) {
    return false;
  }

  // Check if data allows additional properties
  return dataSchema.additionalProperties === true;
}

/**
 * Merges new inputs with existing ones using merge strategy:
 * - Keep all existing inputs unchanged
 * - Add new inputs that don't exist in existing
 */
export function mergeWorkflowInputs(
  existing: Record<string, InputDef>,
  incoming: Record<string, InputDef>,
): Record<string, InputDef> {
  const merged = { ...existing };

  for (const [key, inputDef] of Object.entries(incoming)) {
    // Only add if key doesn't exist in existing
    if (!(key in merged)) {
      merged[key] = inputDef;
    }
  }

  return merged;
}

/**
 * Derives quick options from event_schema.properties.data.properties.
 * Creates a label/path pair for each property in the data schema.
 */
export function deriveQuickOptionsFromSchema(
  eventSchema: JsonObject | null | undefined,
): QuickOption[] {
  const dataSchema = getDataPropertiesFromEventSchema(eventSchema);
  if (!dataSchema) {
    return [];
  }

  const properties = dataSchema.properties as JsonObject;
  const options: QuickOption[] = [];

  for (const key of Object.keys(properties)) {
    // Convert snake_case to Title Case for label
    const label = key
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    options.push({
      label,
      path: `data.${key}`,
    });
  }

  return options;
}

/**
 * Extracts InputDef records from a user-defined JSON Schema (for webhook/form triggers).
 * This handles schemas that users create via the UserSchemaEditor.
 */
export function extractInputsFromUserSchema(
  userSchema: JsonObject | null | undefined,
): Record<string, InputDef> {
  if (!userSchema) {
    return {};
  }

  // User schemas should have structure: { properties: {...} }
  const properties = userSchema.properties as JsonObject | undefined;
  if (!properties || typeof properties !== 'object') {
    return {};
  }

  const requiredFields = Array.isArray(userSchema.required)
    ? (userSchema.required as string[])
    : [];

  const inputs: Record<string, InputDef> = {};

  for (const [key, value] of Object.entries(properties)) {
    if (typeof value !== 'object' || value === null) {
      continue;
    }

    const propSchema = value as JsonObject;
    const schemaType = propSchema.type as JsonSchemaType;

    inputs[key] = {
      type: mapJsonSchemaTypeToInputDefType(schemaType),
      description: (propSchema.description as string) || undefined,
      required: requiredFields.includes(key),
    };
  }

  return inputs;
}

/**
 * Creates a default user schema with common fields for webhook/form triggers.
 */
export function createDefaultUserSchema(): JsonObject {
  return {
    type: 'object',
    properties: {},
    required: [],
    additionalProperties: false,
  };
}

/**
 * Derives quick options from a user-defined schema.
 */
export function deriveQuickOptionsFromUserSchema(
  userSchema: JsonObject | null | undefined,
): QuickOption[] {
  if (!userSchema) {
    return [];
  }

  const properties = userSchema.properties as JsonObject | undefined;
  if (!properties || typeof properties !== 'object') {
    return [];
  }

  const options: QuickOption[] = [];

  for (const key of Object.keys(properties)) {
    const label = key
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    options.push({
      label,
      path: `data.${key}`,
    });
  }

  return options;
}
