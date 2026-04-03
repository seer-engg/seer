/**
 * Simple field-based schema editor.
 * Allows users to add/remove fields with name, type, and description.
 */

import { Plus, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Checkbox } from '@/components/ui/checkbox';
import type { SimpleSchemaEditorProps, SchemaField, SchemaFieldType } from './types';
import { createDefaultField, validateFieldName } from './utils';

const FIELD_TYPES: { value: SchemaFieldType; label: string }[] = [
  { value: 'string', label: 'String' },
  { value: 'integer', label: 'Integer' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Boolean' },
  { value: 'object', label: 'Object' },
  { value: 'array', label: 'Array' },
];

interface FieldRowProps {
  field: SchemaField;
  onUpdate: (updates: Partial<SchemaField>) => void;
  onRemove: () => void;
  showValidation: boolean;
}

function FieldRow({ field, onUpdate, onRemove, showValidation }: FieldRowProps) {
  const nameValidation = showValidation ? validateFieldName(field.name) : { valid: true };

  return (
    <div className="flex items-start gap-2 p-2 rounded-md border bg-muted/30">
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2">
          <Input
            value={field.name}
            onChange={(e) => onUpdate({ name: e.target.value })}
            placeholder="Field name"
            className={`h-8 text-xs flex-1 ${!nameValidation.valid ? 'border-destructive' : ''}`}
          />
          <Select
            value={field.type}
            onValueChange={(value: SchemaFieldType) => onUpdate({ type: value })}
          >
            <SelectTrigger className="h-8 w-24 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FIELD_TYPES.map(({ value, label }) => (
                <SelectItem key={value} value={value} className="text-xs">
                  {label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Input
            value={field.description}
            onChange={(e) => onUpdate({ description: e.target.value })}
            placeholder="Description (optional)"
            className="h-8 text-xs flex-1"
          />
          <div className="flex items-center gap-1.5">
            <Checkbox
              id={`required-${field.id}`}
              checked={field.required}
              onCheckedChange={(checked) => onUpdate({ required: checked === true })}
            />
            <label htmlFor={`required-${field.id}`} className="text-xs text-muted-foreground">
              Required
            </label>
          </div>
        </div>

        {!nameValidation.valid && (
          <p className="text-xs text-destructive">{nameValidation.error}</p>
        )}
      </div>

      <Button
        variant="ghost"
        size="sm"
        onClick={onRemove}
        className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
      >
        <Trash2 className="h-4 w-4" />
      </Button>
    </div>
  );
}

export function SimpleSchemaEditor({ fields, onFieldsChange }: SimpleSchemaEditorProps) {
  const handleAddField = () => {
    onFieldsChange([...fields, createDefaultField()]);
  };

  const handleUpdateField = (index: number, updates: Partial<SchemaField>) => {
    const newFields = [...fields];
    newFields[index] = { ...newFields[index], ...updates };
    onFieldsChange(newFields);
  };

  const handleRemoveField = (index: number) => {
    onFieldsChange(fields.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      {fields.length === 0 ? (
        <div className="text-center py-4 text-sm text-muted-foreground">
          No fields defined. Add a field to define the data schema.
        </div>
      ) : (
        <div className="space-y-2">
          {fields.map((field, index) => (
            <FieldRow
              key={field.id}
              field={field}
              onUpdate={(updates) => handleUpdateField(index, updates)}
              onRemove={() => handleRemoveField(index)}
              showValidation={field.name.length > 0}
            />
          ))}
        </div>
      )}

      <Button
        variant="outline"
        size="sm"
        onClick={handleAddField}
        className="w-full h-8 text-xs"
      >
        <Plus className="h-3.5 w-3.5 mr-1.5" />
        Add Field
      </Button>
    </div>
  );
}
