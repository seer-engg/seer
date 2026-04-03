import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { TemplateConfigField } from '@/types/templates';

export interface TemplateConfigFormProps {
  fields: TemplateConfigField[];
  values: Record<string, unknown>;
  onChange: (name: string, value: unknown) => void;
}

interface FieldProps {
  field: TemplateConfigField;
  value: unknown;
  onChange: (value: unknown) => void;
}

function StringField({ field, value, onChange }: FieldProps) {
  return (
    <Input
      id={field.name}
      value={(value as string) ?? (field.default as string) ?? ''}
      onChange={(e) => onChange(e.target.value)}
      placeholder={field.description}
    />
  );
}

function NumberField({ field, value, onChange }: FieldProps) {
  return (
    <Input
      id={field.name}
      type="number"
      value={(value as number) ?? (field.default as number) ?? ''}
      onChange={(e) => onChange(e.target.value ? Number(e.target.value) : undefined)}
      placeholder={field.description}
    />
  );
}

function BooleanField({ field, value, onChange }: FieldProps) {
  const checked = (value as boolean) ?? (field.default as boolean) ?? false;
  return (
    <div className="flex items-center space-x-2">
      <Checkbox
        id={field.name}
        checked={checked}
        onCheckedChange={(checked) => onChange(checked === true)}
      />
      {field.description && (
        <label
          htmlFor={field.name}
          className="text-sm text-muted-foreground leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
        >
          {field.description}
        </label>
      )}
    </div>
  );
}

function SelectField({ field, value, onChange }: FieldProps) {
  const options = field.options ?? [];
  const currentValue = (value as string) ?? (field.default as string) ?? '';

  return (
    <Select value={currentValue} onValueChange={onChange}>
      <SelectTrigger id={field.name}>
        <SelectValue placeholder={field.description || 'Select an option'} />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/**
 * Dynamic form component that renders configuration fields based on template definition.
 * Supports string, number, boolean, and select field types.
 */
export function TemplateConfigForm({ fields, values, onChange }: TemplateConfigFormProps) {
  if (fields.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This template has no configuration options.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {fields.map((field) => {
        const value = values[field.name];

        return (
          <div key={field.name} className="space-y-2">
            <Label htmlFor={field.name} className="flex items-center gap-1">
              {field.label}
              {field.required && <span className="text-destructive">*</span>}
            </Label>

            {field.type === 'string' && (
              <StringField field={field} value={value} onChange={(v) => onChange(field.name, v)} />
            )}
            {field.type === 'number' && (
              <NumberField field={field} value={value} onChange={(v) => onChange(field.name, v)} />
            )}
            {field.type === 'boolean' && (
              <BooleanField field={field} value={value} onChange={(v) => onChange(field.name, v)} />
            )}
            {field.type === 'select' && (
              <SelectField field={field} value={value} onChange={(v) => onChange(field.name, v)} />
            )}

            {field.description && field.type !== 'boolean' && (
              <p className="text-xs text-muted-foreground">{field.description}</p>
            )}
          </div>
        );
      })}
    </div>
  );
}
