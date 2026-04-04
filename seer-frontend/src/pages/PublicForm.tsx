import { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { toast } from '@/components/ui/sonner';
import { Loader2, CheckCircle2 } from 'lucide-react';
import { getBackendBaseUrl } from '@/lib/api-client';
import { DisplayValue } from '@/components/workflows/executions/DisplayValue';

interface FieldOption {
  value: string;
  label: string;
}

interface FormField {
  name: string;
  displayLabel?: string;
  description?: string;
  type: 'text' | 'number' | 'email' | 'url' | 'object' | 'select' | 'multiselect' | 'checkbox';
  required: boolean;
  placeholder?: string;
  options?: FieldOption[];
}

interface DisplayItem {
  label: string;
  value: string;
}

interface FormConfig {
  form_id: number;
  title: string;
  description?: string;
  fields: FormField[];
  submit_button_text: string;
  success_message: string;
  styling?: Record<string, unknown>;
  /** Display items for HITL forms - show context to the reviewer */
  display_items?: DisplayItem[];
}

/** Display items section for HITL forms */
const DisplayItemsSection = ({ items }: { items: DisplayItem[] }) => (
  <div className="bg-muted/50 rounded-lg p-4 space-y-3 border">
    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">Context</h3>
    <div className="space-y-3">
      {items.map((item, i) => (
        <div key={i} className="flex flex-col sm:flex-row sm:justify-between items-start gap-2 sm:gap-4">
          <span className="text-sm text-muted-foreground shrink-0">{item.label}</span>
          <DisplayValue value={item.value} className="max-w-full sm:max-w-[60%] text-right" />
        </div>
      ))}
    </div>
  </div>
);

/** Select field component */
const SelectField = ({
  field,
  value,
  onChange,
  error,
}: {
  field: FormField;
  value: string;
  onChange: (value: string) => void;
  error?: string;
}) => (
  <Select value={value || ''} onValueChange={onChange}>
    <SelectTrigger className={error ? 'border-destructive' : ''}>
      <SelectValue placeholder={field.placeholder || 'Select an option'} />
    </SelectTrigger>
    <SelectContent>
      {field.options?.map((option) => (
        <SelectItem key={option.value} value={option.value}>
          {option.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);

/** Multi-select field component (checkbox group) */
const MultiSelectField = ({
  field,
  value,
  onChange,
  error,
}: {
  field: FormField;
  value: string[];
  onChange: (value: string[]) => void;
  error?: string;
}) => {
  const handleToggle = (optionValue: string, checked: boolean) => {
    if (checked) {
      onChange([...(value || []), optionValue]);
    } else {
      onChange((value || []).filter((v) => v !== optionValue));
    }
  };

  return (
    <div className={`space-y-2 ${error ? 'text-destructive' : ''}`}>
      {field.options?.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <Checkbox
            id={`${field.name}-${option.value}`}
            checked={(value || []).includes(option.value)}
            onCheckedChange={(checked) => handleToggle(option.value, checked === true)}
          />
          <Label
            htmlFor={`${field.name}-${option.value}`}
            className="text-sm font-normal cursor-pointer"
          >
            {option.label}
          </Label>
        </div>
      ))}
    </div>
  );
};

/** Single checkbox field component */
const CheckboxField = ({
  field,
  value,
  onChange,
}: {
  field: FormField;
  value: boolean;
  onChange: (value: boolean) => void;
}) => (
  <div className="flex items-center space-x-2">
    <Checkbox
      id={field.name}
      checked={value || false}
      onCheckedChange={(checked) => onChange(checked === true)}
    />
    <Label htmlFor={field.name} className="text-sm font-normal cursor-pointer">
      {field.displayLabel || field.name}
    </Label>
  </div>
);

/** Renders the appropriate input control based on field type */
const FieldControl = ({
  field,
  value,
  onChange,
  onMultiChange,
  error,
}: {
  field: FormField;
  value: unknown;
  onChange: (value: string) => void;
  onMultiChange: (value: string[]) => void;
  error?: string;
}) => {
  switch (field.type) {
    case 'select':
      return <SelectField field={field} value={value as string} onChange={onChange} error={error} />;
    case 'multiselect':
      return (
        <MultiSelectField
          field={field}
          value={value as string[]}
          onChange={onMultiChange}
          error={error}
        />
      );
    case 'object':
      return (
        <Textarea
          id={field.name}
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder}
          required={field.required}
          className={error ? 'border-destructive' : ''}
        />
      );
    default:
      return (
        <Input
          id={field.name}
          type={field.type}
          value={(value as string) || ''}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder}
          required={field.required}
          className={error ? 'border-destructive' : ''}
        />
      );
  }
};

// Helper: Render form field
const FormFieldInput = ({
  field,
  value,
  onChange,
  onMultiChange,
  onBoolChange,
  error,
}: {
  field: FormField;
  value: unknown;
  onChange: (value: string) => void;
  onMultiChange: (value: string[]) => void;
  onBoolChange: (value: boolean) => void;
  error?: string;
}) => {
  if (field.type === 'checkbox') {
    return (
      <div className="space-y-2">
        {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
        <CheckboxField field={field} value={value as boolean} onChange={onBoolChange} />
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <Label htmlFor={field.name}>
        {field.displayLabel || field.name}
        {field.required && <span className="text-destructive ml-1">*</span>}
      </Label>
      {field.description && <p className="text-xs text-muted-foreground">{field.description}</p>}
      <FieldControl
        field={field}
        value={value}
        onChange={onChange}
        onMultiChange={onMultiChange}
        error={error}
      />
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
};

// Helper: Form submission hook
const useFormSubmission = (formConfig: FormConfig | null, suffix: string | undefined) => {
  const [formData, setFormData] = useState<Record<string, unknown>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formConfig) return;

    const newErrors: Record<string, string> = {};
    formConfig.fields.forEach((field) => {
      const fieldValue = formData[field.name];
      const isEmpty =
        fieldValue === undefined ||
        fieldValue === null ||
        fieldValue === '' ||
        (Array.isArray(fieldValue) && fieldValue.length === 0);

      if (field.required && isEmpty) {
        newErrors[field.name] = `${field.displayLabel || field.name} is required`;
      }
    });

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsSubmitting(true);
    try {
      const res = await fetch(`${getBackendBaseUrl()}/api/forms/submit/${suffix}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const error = await res.json();
        if (error.detail?.errors) {
          const fieldErrors: Record<string, string> = {};
          error.detail.errors.forEach((err: string) => {
            fieldErrors.general = err;
          });
          setErrors(fieldErrors);
          toast.error('Please fix the errors below');
          return;
        }
        throw new Error('Submission failed');
      }

      setIsSubmitted(true);
      toast.success(formConfig.success_message || 'Submitted!');
    } catch {
      toast.error('Submission failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const updateField = (name: string, value: unknown) => {
    setFormData({ ...formData, [name]: value });
    setErrors({ ...errors, [name]: '' });
  };

  return { formData, isSubmitting, isSubmitted, errors, handleSubmit, updateField };
};

export default function PublicForm() {
  const { suffix } = useParams<{ suffix: string }>();
  const [formConfig, setFormConfig] = useState<FormConfig | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { formData, isSubmitting, isSubmitted, errors, handleSubmit, updateField } =
    useFormSubmission(formConfig, suffix);

  const loadForm = useCallback(async () => {
    try {
      const res = await fetch(`${getBackendBaseUrl()}/api/forms/resolve/${suffix}`);
      if (!res.ok) throw new Error('Form not found');
      const data = await res.json();
      setFormConfig(data);
    } catch {
      toast.error('Form not found');
    } finally {
      setIsLoading(false);
    }
  }, [suffix]);

  useEffect(() => {
    loadForm();
  }, [loadForm]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!formConfig) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div className="text-center">
          <h1 className="text-2xl font-bold">Form Not Found</h1>
          <p className="mt-2 text-muted-foreground">
            The form you're looking for doesn't exist or has been disabled.
          </p>
        </div>
      </div>
    );
  }

  if (isSubmitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <div className="text-center space-y-4">
          <CheckCircle2 className="h-16 w-16 text-success mx-auto" />
          <h2 className="text-2xl font-bold">{formConfig.success_message || 'Thank You!'}</h2>
        </div>
      </div>
    );
  }

  const hasDisplayItems = formConfig.display_items && formConfig.display_items.length > 0;

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="max-w-2xl w-full">
        <div className="bg-card border rounded-lg p-8 space-y-6">
          <div>
            <h1 className="text-3xl font-bold">{formConfig.title}</h1>
            {formConfig.description && (
              <p className="text-muted-foreground mt-2">{formConfig.description}</p>
            )}
          </div>

          {/* Display items section for HITL forms */}
          {hasDisplayItems && <DisplayItemsSection items={formConfig.display_items!} />}

          <form onSubmit={handleSubmit} className="space-y-4">
            {formConfig.fields.map((field) => (
              <FormFieldInput
                key={field.name}
                field={field}
                value={formData[field.name]}
                onChange={(value) => updateField(field.name, value)}
                onMultiChange={(value) => updateField(field.name, value)}
                onBoolChange={(value) => updateField(field.name, value)}
                error={errors[field.name]}
              />
            ))}
            <Button type="submit" disabled={isSubmitting} className="w-full">
              {isSubmitting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Submitting...
                </>
              ) : (
                formConfig.submit_button_text || 'Submit'
              )}
            </Button>
          </form>
          <div className="text-center text-xs text-muted-foreground pt-4 border-t">
            Powered by <span className="font-semibold">Seer</span>
          </div>
        </div>
      </div>
    </div>
  );
}
