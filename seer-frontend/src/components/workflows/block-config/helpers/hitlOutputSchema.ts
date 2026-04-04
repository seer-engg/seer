import type { HitlInputField } from '../types';

/**
 * Maps HITL input_type to JSON Schema type
 */
function inputTypeToJsonSchemaType(inputType: HitlInputField['input_type']): Record<string, unknown> {
  switch (inputType) {
    case 'text':
      return { type: 'string' };
    case 'number':
      return { type: 'number' };
    case 'boolean':
      return { type: 'boolean' };
    case 'single_choice':
      return { type: 'string' };
    case 'multi_choice':
      return { type: 'array', items: { type: 'string' } };
    case 'table':
      return { type: 'array', items: { type: 'object' } };
    default:
      return { type: 'string' };
  }
}

/**
 * Applies enum constraints to a schema for choice-type fields
 */
function applyChoiceEnums(
  schema: Record<string, unknown>,
  inputType: string,
  options: Array<{ value: string }> | undefined
): void {
  if (!options?.length) return;
  const enumValues = options.map(opt => opt.value).filter(Boolean);
  if (inputType === 'single_choice') {
    schema.enum = enumValues;
  } else if (inputType === 'multi_choice') {
    (schema.items as Record<string, unknown>).enum = enumValues;
  }
}

/**
 * Builds a JSON Schema for a single HITL field, including nested table columns
 */
function buildFieldSchema(input: { input_type: string; options?: Array<{ value: string }>; columns?: HitlInputField['columns'] }): Record<string, unknown> {
  const schema = inputTypeToJsonSchemaType(input.input_type as HitlInputField['input_type']);
  applyChoiceEnums(schema, input.input_type, input.options);

  if (input.input_type === 'table' && input.columns?.length) {
    const colProperties: Record<string, Record<string, unknown>> = {};
    for (const col of input.columns) {
      const colSchema = inputTypeToJsonSchemaType(col.input_type);
      applyChoiceEnums(colSchema, col.input_type, col.options);
      colProperties[col.id] = colSchema;
    }
    schema.items = { type: 'object', properties: colProperties };
  }

  return schema;
}

/**
 * Generates a JSON Schema from HITL input field definitions.
 * Each input field's id becomes a property in the output schema.
 *
 * This enables variable suggestions like {{hitl-1.approval}} in downstream nodes.
 *
 * @example
 * ```ts
 * const inputs = [
 *   { id: 'approval', input_type: 'single_choice', options: [{value: 'yes'}, {value: 'no'}] },
 *   { id: 'reason', input_type: 'text' }
 * ];
 * generateHitlOutputSchema(inputs);
 * // Returns:
 * // {
 * //   type: 'object',
 * //   properties: {
 * //     approval: { type: 'string', enum: ['yes', 'no'] },
 * //     reason: { type: 'string' }
 * //   }
 * // }
 * ```
 */
export function generateHitlOutputSchema(
  inputs: HitlInputField[] | undefined
): Record<string, unknown> | undefined {
  if (!inputs || inputs.length === 0) {
    return undefined;
  }

  const properties: Record<string, Record<string, unknown>> = {};

  for (const input of inputs) {
    if (!input.id) continue;
    properties[input.id] = buildFieldSchema(input);
  }

  if (Object.keys(properties).length === 0) {
    return undefined;
  }

  return {
    type: 'object',
    properties,
  };
}
