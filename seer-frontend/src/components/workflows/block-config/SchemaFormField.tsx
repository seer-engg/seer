import { useCallback, useState } from 'react';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { InfoIcon } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Tooltip, TooltipTrigger, TooltipContent } from '@/components/ui/tooltip';

export interface ParameterSchema {
  type?: string; // 'string', 'integer', 'number', 'boolean', 'array', 'object'
  description?: string;
  default?: import('@/types/workflow-spec').JsonValue;
  enum?: string[]; // For select dropdowns
  minimum?: number;
  maximum?: number;
  required?: boolean;
  [key: string]: unknown; // Allow additional schema properties
}

export interface SchemaFormFieldProps {
  name: string;
  schema: ParameterSchema;
  value: import('@/types/workflow-spec').JsonValue;
  onChange: (value: import('@/types/workflow-spec').JsonValue) => void;
  required?: boolean;
  compact?: boolean; // Use ultra-compact spacing
  className?: string;
  error?: string; // Validation error message
}

// Helper: Get spacing config
const getSpacing = (compact: boolean) =>
  compact
    ? { container: 'space-y-1', labelText: 'text-xs', inputHeight: 'h-9', gap: 'gap-2' }
    : { container: 'space-y-2', labelText: 'text-sm', inputHeight: 'h-10', gap: 'gap-3' };

// Helper: Render field label with optional tooltip
const FieldLabel = ({ name, required, schema, spacing }: {
  name: string;
  required: boolean;
  schema: ParameterSchema;
  spacing: ReturnType<typeof getSpacing>;
}) => (
  <div className="flex items-center gap-1.5">
    <Label htmlFor={name} className={cn(spacing.labelText, 'font-medium')}>
      {name}
      {required && <span className="text-destructive ml-0.5">*</span>}
    </Label>
    {schema.description && (
      <Tooltip>
        <TooltipTrigger asChild>
          <InfoIcon className="h-3 w-3 text-muted-foreground cursor-help" />
        </TooltipTrigger>
        <TooltipContent side="right" className="max-w-xs">
          <p className="text-xs">{schema.description}</p>
          {schema.default !== undefined && (
            <p className="text-xs text-muted-foreground mt-1">
              Default: {JSON.stringify(schema.default)}
            </p>
          )}
        </TooltipContent>
      </Tooltip>
    )}
  </div>
);

// Helper: Render number input
const renderNumberInput = (opts: {
  type: 'integer' | 'number';
  name: string;
  value: import('@/types/workflow-spec').JsonValue;
  schema: ParameterSchema;
  spacing: ReturnType<typeof getSpacing>;
  showError: boolean;
  handleChange: (v: import('@/types/workflow-spec').JsonValue) => void;
  handleBlur: () => void;
}) => (
  <Input
    type="number"
    id={opts.name}
    value={opts.value ?? ''}
    onChange={(e) => {
      const val = e.target.value;
      if (!val) opts.handleChange('');
      else {
        const num = opts.type === 'integer' ? parseInt(val, 10) : parseFloat(val);
        opts.handleChange(isNaN(num) ? val : num);
      }
    }}
    onBlur={opts.handleBlur}
    placeholder={opts.schema.default !== undefined ? String(opts.schema.default) : undefined}
    min={opts.schema.minimum}
    max={opts.schema.maximum}
    className={cn(opts.spacing.inputHeight, opts.spacing.labelText, opts.showError && 'border-destructive')}
  />
);

// Helper: Render JSON input (array/object)
const renderJsonInput = (opts: {
  type: 'array' | 'object';
  name: string;
  value: import('@/types/workflow-spec').JsonValue;
  schema: ParameterSchema;
  spacing: ReturnType<typeof getSpacing>;
  showError: boolean;
  handleChange: (v: import('@/types/workflow-spec').JsonValue) => void;
  handleBlur: () => void;
}) => {
  const placeholder = opts.type === 'array' ? '[\n  "value1",\n  "value2"\n]' : '{\n  "key": "value"\n}';
  const isValid = (p: unknown) => opts.type === 'array' ? Array.isArray(p) : typeof p === 'object' && !Array.isArray(p);
  return (
    <Textarea
      id={opts.name}
      value={typeof opts.value === 'string' ? opts.value : JSON.stringify(opts.value, null, 2)}
      onChange={(e) => {
        try {
          const parsed = JSON.parse(e.target.value);
          opts.handleChange(isValid(parsed) ? parsed : e.target.value);
        } catch {
          opts.handleChange(e.target.value);
        }
      }}
      onBlur={opts.handleBlur}
      placeholder={opts.schema.default ? JSON.stringify(opts.schema.default, null, 2) : placeholder}
      rows={3}
      className={cn('max-h-[120px] resize-none', opts.spacing.labelText, opts.showError && 'border-destructive')}
    />
  );
};

// Helper: Render input field based on schema type
const renderFieldInput = (opts: {
  schema: ParameterSchema;
  value: import('@/types/workflow-spec').JsonValue;
  name: string;
  spacing: ReturnType<typeof getSpacing>;
  showError: boolean;
  handleChange: (v: import('@/types/workflow-spec').JsonValue) => void;
  handleBlur: () => void;
}) => {
  const type = opts.schema.type || 'string';

  if (type === 'boolean') {
    return (
      <div className="flex items-center gap-2">
        <Checkbox id={opts.name} checked={!!opts.value} onCheckedChange={opts.handleChange} onBlur={opts.handleBlur} />
        <Label htmlFor={opts.name} className="font-normal cursor-pointer">{opts.name}</Label>
      </div>
    );
  }

  if (opts.schema.enum && opts.schema.enum.length > 0) {
    return (
      <Select value={String(opts.value || '')} onValueChange={opts.handleChange}>
        <SelectTrigger className={cn(opts.spacing.inputHeight, opts.showError && 'border-destructive')}>
          <SelectValue placeholder={`Select ${opts.name}`} />
        </SelectTrigger>
        <SelectContent>
          {opts.schema.enum.map((option) => (
            <SelectItem key={option} value={option}>{option}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  if (type === 'integer' || type === 'number') return renderNumberInput({ ...opts, type });
  if (type === 'array' || type === 'object') return renderJsonInput({ ...opts, type });

  return (
    <Input
      type="text"
      id={opts.name}
      value={opts.value ?? ''}
      onChange={(e) => opts.handleChange(e.target.value)}
      onBlur={opts.handleBlur}
      placeholder={opts.schema.default !== undefined ? String(opts.schema.default) : undefined}
      className={cn(opts.spacing.inputHeight, opts.spacing.labelText, opts.showError && 'border-destructive')}
    />
  );
};

export function SchemaFormField({
  name,
  schema,
  value,
  onChange,
  required = false,
  compact = true,
  className,
  error,
}: SchemaFormFieldProps) {
  const [touched, setTouched] = useState(false);
  const spacing = getSpacing(compact);
  const showError = touched && error;
  const handleBlur = () => setTouched(true);
  const handleChange = useCallback(
    (newValue: import('@/types/workflow-spec').JsonValue) => onChange(newValue),
    [onChange]
  );

  const shouldShowLabel = schema.type !== 'boolean';

  return (
    <div className={cn(spacing.container, className)}>
      {shouldShowLabel && <FieldLabel name={name} required={required} schema={schema} spacing={spacing} />}
      {renderFieldInput({ schema, value, name, spacing, showError, handleChange, handleBlur })}
      {showError && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
