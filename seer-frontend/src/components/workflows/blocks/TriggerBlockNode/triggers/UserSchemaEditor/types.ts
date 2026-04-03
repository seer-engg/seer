/**
 * Types for the UserSchemaEditor component.
 */

import type { JsonObject } from '@/types/workflow-spec';

export type SchemaFieldType = 'string' | 'integer' | 'number' | 'boolean' | 'object' | 'array';

export interface SchemaField {
  id: string;
  name: string;
  type: SchemaFieldType;
  description: string;
  required: boolean;
}

export type SchemaEditorMode = 'simple' | 'advanced';

export interface UserSchemaEditorProps {
  schema: JsonObject;
  onChange: (schema: JsonObject) => void;
  mode?: SchemaEditorMode;
  onModeChange?: (mode: SchemaEditorMode) => void;
}

export interface SimpleSchemaEditorProps {
  fields: SchemaField[];
  onFieldsChange: (fields: SchemaField[]) => void;
}

export interface AdvancedSchemaEditorProps {
  schema: JsonObject;
  onSchemaChange: (schema: JsonObject) => void;
  error?: string;
}
