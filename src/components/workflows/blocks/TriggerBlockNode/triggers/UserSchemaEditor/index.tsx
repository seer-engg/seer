/**
 * User Schema Editor for webhook/form triggers.
 * Provides simple (field-based) mode that can be converted to advanced (JSON Schema) mode.
 * Conversion is one-way: simple → advanced only.
 */

import { useState, useCallback } from 'react';
import { Code2, ArrowRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { JsonObject } from '@/types/workflow-spec';
import type { UserSchemaEditorProps, SchemaEditorMode, SchemaField } from './types';
import { SimpleSchemaEditor } from './SimpleSchemaEditor';
import { AdvancedSchemaEditor } from './AdvancedSchemaEditor';
import { fieldsToJsonSchema, jsonSchemaToFields } from './utils';

export function UserSchemaEditor({
  schema,
  onChange,
}: Omit<UserSchemaEditorProps, 'mode' | 'onModeChange'>) {
  const [mode, setMode] = useState<SchemaEditorMode>('simple');
  const [fields, setFields] = useState<SchemaField[]>(() => jsonSchemaToFields(schema));

  const handleFieldsChange = useCallback(
    (newFields: SchemaField[]) => {
      setFields(newFields);
      // Auto-update schema as fields change (for live preview/validation)
      onChange(fieldsToJsonSchema(newFields));
    },
    [onChange],
  );

  const handleGenerateAndSwitch = useCallback(() => {
    const generatedSchema = fieldsToJsonSchema(fields);
    onChange(generatedSchema);
    setMode('advanced');
  }, [fields, onChange]);

  const handleSchemaChange = useCallback(
    (newSchema: JsonObject) => {
      onChange(newSchema);
    },
    [onChange],
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
          Data Schema
        </div>
        {mode === 'simple' ? (
          <Button
            onClick={handleGenerateAndSwitch}
            size="sm"
            variant="outline"
            className="h-7 px-2 text-xs gap-1.5"
          >
            <Code2 className="h-3.5 w-3.5" />
            Generate & Switch to Advanced
            <ArrowRight className="h-3 w-3" />
          </Button>
        ) : (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-muted text-xs text-muted-foreground">
            <Code2 className="h-3.5 w-3.5" />
            Advanced Mode
          </div>
        )}
      </div>

      {mode === 'simple' ? (
        <>
          <SimpleSchemaEditor
            fields={fields}
            onFieldsChange={handleFieldsChange}
          />
          <p className="text-xs text-muted-foreground">
            Add fields for your webhook/form data. Click "Generate & Switch to Advanced" to generate JSON Schema and unlock full editing.
          </p>
        </>
      ) : (
        <>
          <AdvancedSchemaEditor
            schema={schema}
            onSchemaChange={handleSchemaChange}
          />
          <p className="text-xs text-muted-foreground">
            Edit the JSON Schema directly. Changes save automatically when you click outside the editor.
          </p>
        </>
      )}
    </div>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export { createEmptySchema } from './utils';
export type { SchemaEditorMode, SchemaField } from './types';
