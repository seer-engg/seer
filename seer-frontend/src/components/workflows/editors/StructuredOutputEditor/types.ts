/**
 * Types for the recursive StructuredOutputEditor.
 */

/** Recursive JSON Schema node */
export interface JsonSchemaNode {
  type: string;
  description?: string;
  properties?: Record<string, JsonSchemaNode>;
  required?: string[];
  items?: JsonSchemaNode;
  minimum?: number;
  maximum?: number;
}

/** Top-level output schema — always an object */
export interface JsonSchema extends JsonSchemaNode {
  type: 'object';
  properties: Record<string, JsonSchemaNode>;
}

/**
 * Recursive field definition used internally by the visual editor.
 * `id` is a stable React key and is NOT emitted to the JSON Schema.
 */
export interface FieldDefinition {
  id: string;
  name: string;
  /** Pydantic-style type: 'str' | 'int' | 'float' | 'bool' | 'list' | 'dict' | 'any' */
  type: string;
  description?: string;
  /** Sub-properties for dict/object fields */
  children?: FieldDefinition[];
  /** Primitive type of list items, or 'object' for structured items */
  itemType?: string;
  /** Sub-properties for list-of-object item fields */
  itemChildren?: FieldDefinition[];
}

export interface StructuredOutputEditorProps {
  value?: JsonSchema | Record<string, unknown>;
  onChange: (schema: JsonSchema) => void;
}
