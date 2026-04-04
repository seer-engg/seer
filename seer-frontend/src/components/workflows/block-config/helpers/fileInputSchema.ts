/**
 * FILE_INPUT_SCHEMA detection and type utilities
 *
 * The backend uses a standardized FILE_INPUT_SCHEMA with a oneOf structure:
 * - workflow_file_ref: References file output from an upstream workflow node
 * - static_file_ref: References a file from user's storage
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Reference to a file output from an upstream workflow node
 */
export interface WorkflowFileRef {
  _type: 'workflow_file_ref';
  source_node_id: string;
  output_field: string;
}

/**
 * Reference to a static file from user's storage
 */
export interface StaticFileRef {
  _type: 'static_file_ref';
  file_id: string;
}

/**
 * Union type for file input values
 * - string: Template expression like "${nodeId.field}" for workflow file references
 * - StaticFileRef: Reference to a static file from user's storage
 * - null: No value set
 */
export type FileInputValue = string | StaticFileRef | null;

// ============================================================================
// Type Guards
// ============================================================================

/**
 * Check if a value is a WorkflowFileRef
 */
export function isWorkflowFileRef(value: unknown): value is WorkflowFileRef {
  if (!value || typeof value !== 'object') return false;
  const obj = value as Record<string, unknown>;
  return (
    obj._type === 'workflow_file_ref' &&
    typeof obj.source_node_id === 'string' &&
    typeof obj.output_field === 'string'
  );
}

/**
 * Check if a value is a StaticFileRef
 */
export function isStaticFileRef(value: unknown): value is StaticFileRef {
  if (!value || typeof value !== 'object') return false;
  const obj = value as Record<string, unknown>;
  return obj._type === 'static_file_ref' && typeof obj.file_id === 'string';
}

/**
 * Check if a value is any valid FileInputValue (excluding null)
 */
export function isFileInputValue(value: unknown): value is string | StaticFileRef {
  return isTemplateExpression(value) || isStaticFileRef(value);
}

/**
 * Check if a value is a template expression like "${nodeId.field}"
 */
export function isTemplateExpression(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  return /^\$\{[^}]+\}$/.test(value.trim());
}

// ============================================================================
// Schema Detection
// ============================================================================

interface SchemaWithOneOf {
  oneOf?: unknown[];
  [key: string]: unknown;
}

interface SchemaOption {
  type?: string;
  properties?: {
    _type?: {
      const?: string;
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  [key: string]: unknown;
}

/**
 * Check if a schema matches FILE_INPUT_SCHEMA pattern
 *
 * FILE_INPUT_SCHEMA has a oneOf with two options:
 * 1. workflow_file_ref with source_node_id and output_field
 * 2. static_file_ref with file_id
 */
export function isFileInputSchema(schema: unknown): boolean {
  if (!schema || typeof schema !== 'object') return false;

  const schemaObj = schema as SchemaWithOneOf;
  const oneOf = schemaObj.oneOf;

  if (!Array.isArray(oneOf) || oneOf.length < 2) return false;

  let hasWorkflowRef = false;
  let hasStaticRef = false;

  for (const option of oneOf) {
    if (!option || typeof option !== 'object') continue;

    const opt = option as SchemaOption;
    const typeConst = opt.properties?._type?.const;

    if (typeConst === 'workflow_file_ref') {
      hasWorkflowRef = true;
    } else if (typeConst === 'static_file_ref') {
      hasStaticRef = true;
    }
  }

  return hasWorkflowRef && hasStaticRef;
}

interface ArraySchema {
  type?: string;
  items?: unknown;
  [key: string]: unknown;
}

/**
 * Check if a schema is an array of FILE_INPUT_SCHEMA
 * Used for gmail attachments: attachments[].file
 */
export function isFileInputArraySchema(schema: unknown): boolean {
  if (!schema || typeof schema !== 'object') return false;

  const schemaObj = schema as ArraySchema;

  if (schemaObj.type !== 'array' || !schemaObj.items) return false;

  return isFileInputSchema(schemaObj.items);
}

// ============================================================================
// Expression Parsing
// ============================================================================

/**
 * Parse an expression like "${nodeId.field}" or "{{nodeId.field}}" to WorkflowFileRef
 * Returns null if expression doesn't match expected pattern
 */
export function parseExpressionToWorkflowRef(expression: string): WorkflowFileRef | null {
  if (!expression || typeof expression !== 'string') return null;

  // Match both {{nodeId.field}} (mustache) and ${nodeId.field} (dollar) formats
  // - ^(?:\{\{|\$\{?) - Start with either {{ OR $ with optional {
  // - ([^.}]+) - Capture nodeId (stops at . or })
  // - \. - Literal dot separator
  // - ([^}]+?) - Capture outputField (non-greedy)
  // - (?:\}\}|\}?)$ - End with either }} OR optional single }
  const match = expression.match(/^(?:\{\{|\$\{?)([^.}]+)\.([^}]+?)(?:\}\}|\}?)$/);

  if (!match) return null;

  const [, sourceNodeId, outputField] = match;

  if (!sourceNodeId || !outputField) return null;

  return {
    _type: 'workflow_file_ref',
    source_node_id: sourceNodeId.trim(),
    output_field: outputField.trim(),
  };
}

/**
 * Convert a WorkflowFileRef to expression string
 */
export function workflowRefToExpression(ref: WorkflowFileRef): string {
  return `\${${ref.source_node_id}.${ref.output_field}}`;
}

// ============================================================================
// Value Helpers
// ============================================================================

/**
 * Get display text for a FileInputValue
 */
export function getFileInputDisplayText(
  value: FileInputValue,
  fileInfo?: { filename: string; size_human?: string },
): string {
  if (!value) return '';

  // Template expression string - return as-is
  if (typeof value === 'string') {
    return value;
  }

  if (isStaticFileRef(value)) {
    if (fileInfo) {
      return fileInfo.size_human
        ? `${fileInfo.filename} (${fileInfo.size_human})`
        : fileInfo.filename;
    }
    return value.file_id;
  }

  return '';
}

/**
 * Create a StaticFileRef from a file_id
 */
export function createStaticFileRef(fileId: string): StaticFileRef {
  return {
    _type: 'static_file_ref',
    file_id: fileId,
  };
}

/**
 * Create a WorkflowFileRef from source node and field
 */
export function createWorkflowFileRef(sourceNodeId: string, outputField: string): WorkflowFileRef {
  return {
    _type: 'workflow_file_ref',
    source_node_id: sourceNodeId,
    output_field: outputField,
  };
}
