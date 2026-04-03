/**
 * Structured Output Editor
 *
 * Visual (recursive) + raw JSON editor for defining LLM structured output schemas.
 * Supports nested object and list-of-object field definitions up to 3 levels deep.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Plus, Code2 } from 'lucide-react';
import { FieldRow } from './FieldRow';
import { RawSchemaEditor } from './RawSchemaEditor';
import { schemaToFields, fieldsToSchema, createEmptyField } from './utils';
import type { FieldDefinition, JsonSchema, StructuredOutputEditorProps } from './types';

const DEFAULT_SCHEMA: JsonSchema = { type: 'object', properties: {} };

// ─── Sub-components ───────────────────────────────────────────────────────────

interface VisualFieldListProps {
  fields: FieldDefinition[];
  onUpdate: (index: number, updates: Partial<FieldDefinition>) => void;
  onRemove: (index: number) => void;
}

function VisualFieldList({ fields, onUpdate, onRemove }: VisualFieldListProps) {
  if (fields.length === 0) {
    return (
      <div className="rounded-lg border p-4 text-center text-muted-foreground text-sm">
        No fields. Click &quot;Add Field&quot; to get started.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {fields.map((field, index) => (
        <FieldRow
          key={field.id}
          field={field}
          index={index}
          depth={0}
          canRemove={fields.length > 1}
          onUpdate={onUpdate}
          onRemove={onRemove}
        />
      ))}
    </div>
  );
}

interface EditorContentProps {
  mode: 'visual' | 'raw';
  schema: JsonSchema;
  fields: FieldDefinition[];
  onRawChange: (s: JsonSchema) => void;
  onUpdate: (i: number, u: Partial<FieldDefinition>) => void;
  onRemove: (i: number) => void;
}

function EditorContent({ mode, schema, fields, onRawChange, onUpdate, onRemove }: EditorContentProps) {
  if (mode === 'raw') {
    return <RawSchemaEditor schema={schema} onChange={onRawChange} />;
  }
  return <VisualFieldList fields={fields} onUpdate={onUpdate} onRemove={onRemove} />;
}

// ─── Main component ───────────────────────────────────────────────────────────

export function StructuredOutputEditor({ value, onChange }: StructuredOutputEditorProps) {
  const [mode, setMode] = useState<'visual' | 'raw'>('visual');
  const [fields, setFields] = useState<FieldDefinition[]>([createEmptyField()]);
  const lastEmittedRef = useRef<string>('');

  // Sync from parent → internal state, skip round-trips we emitted ourselves
  useEffect(() => {
    const hasValue = value && Object.keys(value).length > 0;
    const serialized = JSON.stringify(hasValue ? value : DEFAULT_SCHEMA);
    if (serialized === lastEmittedRef.current) return;
    if (hasValue) {
      const parsed = schemaToFields(value);
      setFields(parsed.length > 0 ? parsed : [createEmptyField()]);
    } else {
      setFields([createEmptyField()]);
      onChange(DEFAULT_SCHEMA);
    }
    lastEmittedRef.current = serialized;
  }, [value, onChange]);

  const handleFieldChange = useCallback(
    (updater: (prev: FieldDefinition[]) => FieldDefinition[]) => {
      setFields((prev) => {
        const next = updater(prev);
        const schema = fieldsToSchema(next);
        lastEmittedRef.current = JSON.stringify(schema);
        onChange(schema);
        return next.length > 0 ? next : [createEmptyField()];
      });
    },
    [onChange],
  );

  const handleUpdate = useCallback(
    (i: number, u: Partial<FieldDefinition>) =>
      handleFieldChange((prev) => { const n = [...prev]; n[i] = { ...n[i], ...u }; return n; }),
    [handleFieldChange],
  );

  const handleRemove = useCallback(
    (i: number) => handleFieldChange((prev) => prev.filter((_, idx) => idx !== i)),
    [handleFieldChange],
  );

  const handleRawChange = useCallback(
    (s: JsonSchema) => { lastEmittedRef.current = JSON.stringify(s); onChange(s); },
    [onChange],
  );

  const handleModeToggle = useCallback(() => {
    setMode((prev) => {
      if (prev === 'raw' && value) {
        const parsed = schemaToFields(value as JsonSchema);
        setFields(parsed.length > 0 ? parsed : [createEmptyField()]);
      }
      return prev === 'visual' ? 'raw' : 'visual';
    });
  }, [value]);

  const currentSchema = (value as JsonSchema) ?? DEFAULT_SCHEMA;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Label>Output Fields</Label>
        <div className="flex items-center gap-2">
          <Button type="button" variant="outline" size="sm" onClick={handleModeToggle}
            title={mode === 'visual' ? 'Switch to raw JSON editor' : 'Switch to visual editor'}>
            <Code2 className="w-4 h-4" />
          </Button>
          {mode === 'visual' && (
            <Button type="button" variant="outline" size="sm"
              onClick={() => handleFieldChange((prev) => [...prev, createEmptyField()])}>
              <Plus className="w-4 h-4 mr-1" />
              Add Field
            </Button>
          )}
        </div>
      </div>

      <EditorContent
        mode={mode}
        schema={currentSchema}
        fields={fields}
        onRawChange={handleRawChange}
        onUpdate={handleUpdate}
        onRemove={handleRemove}
      />

      <p className="text-xs text-muted-foreground">
        Define the structure of the LLM output. Use Dictionary or List types for nested data.
        Switch to raw JSON mode for full schema control.
      </p>
    </div>
  );
}
