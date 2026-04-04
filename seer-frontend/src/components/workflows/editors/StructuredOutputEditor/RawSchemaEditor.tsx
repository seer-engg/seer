/**
 * Raw JSON Schema editor — adapted from AdvancedSchemaEditor.tsx.
 * Changes are validated and saved on blur.
 */

import { useCallback, useState, useEffect } from 'react';
import { AlertCircle, Check } from 'lucide-react';
import type { JsonSchema } from './types';
import { validateJsonSchema } from './utils';

interface RawSchemaEditorProps {
  schema: JsonSchema;
  onChange: (schema: JsonSchema) => void;
  error?: string | null;
}

export function RawSchemaEditor({ schema, onChange, error: externalError }: RawSchemaEditorProps) {
  const [localValue, setLocalValue] = useState(() => JSON.stringify(schema, null, 2));
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isDirty, setIsDirty] = useState(false);

  // Sync localValue when schema changes externally (e.g. mode switch)
  useEffect(() => {
    if (!isDirty) {
      setLocalValue(JSON.stringify(schema, null, 2));
    }
  }, [schema, isDirty]);

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setLocalValue(e.target.value);
    setIsDirty(true);
    setValidationError(null);
  }, []);

  const handleBlur = useCallback(() => {
    if (!isDirty) return;

    const result = validateJsonSchema(localValue);
    if (result.valid) {
      onChange(result.schema);
      setValidationError(null);
      setIsDirty(false);
    } else if (!result.valid) {
      setValidationError(result.error);
    }
  }, [localValue, isDirty, onChange]);

  const displayError = externalError || validationError;
  const isValid = !displayError && !isDirty;

  return (
    <div className="space-y-2">
      <div className="relative">
        <textarea
          value={localValue}
          onChange={handleChange}
          onBlur={handleBlur}
          className={`
            w-full h-64 p-3 text-xs font-mono rounded-md border bg-background
            resize-none focus:outline-none focus:ring-2 focus:ring-ring
            ${displayError ? 'border-destructive focus:ring-destructive' : ''}
          `}
          placeholder={`{
  "type": "object",
  "properties": {
    "field_name": {
      "type": "string",
      "description": "Field description"
    }
  }
}`}
          spellCheck={false}
        />

        <div className="absolute top-2 right-2">
          {isValid && (
            <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-success/10 text-success text-xs">
              <Check className="h-3 w-3" />
              Saved
            </div>
          )}
          {isDirty && !displayError && (
            <div className="px-2 py-1 rounded-md bg-warning/10 text-warning text-xs">
              Unsaved changes
            </div>
          )}
        </div>
      </div>

      {displayError && (
        <div className="flex items-start gap-2 p-2 rounded-md bg-destructive/10 text-destructive text-xs">
          <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
          <span>{displayError}</span>
        </div>
      )}
    </div>
  );
}
