import { FormField } from './FormField';
import type { TemplateAutocompleteControls, ResourcePickerConfig } from '../types';
import {
  ResourcePickerField,
  EnumSelectField,
  BooleanField,
  NumericField,
  JsonField,
  TextField,
  RichTextField,
  FileInputField,
  FileInputArrayField,
  CronScheduleField,
} from './fields';
import {
  isFileInputSchema,
  isFileInputArraySchema,
  type FileInputValue,
} from '../helpers/fileInputSchema';
import type { FileVariable } from '../helpers/discoverFileVariables';

export type DynamicFieldDef = {
  type?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  multiline?: boolean;
  [key: string]: unknown;
};

export interface DynamicFormFieldProps {
  name: string;
  label?: string;
  description?: string;
  required?: boolean;
  defaultValue?: unknown;
  value: unknown;
  onChange: (value: unknown) => void;
  def?: DynamicFieldDef;
  templateAutocomplete: TemplateAutocompleteControls;
  provider?: string;
  dependsOnValues?: Record<string, string>;
  placeholder?: string;
  rows?: number;
  className?: string;
  error?: string;
  onResourceLabelChange?: (fieldName: string, label?: string) => void;
  /** Available file variables from upstream nodes (for FILE_INPUT_SCHEMA fields) */
  availableFileVariables?: FileVariable[];
}

// eslint-disable-next-line max-lines-per-function
export function DynamicFormField(props: DynamicFormFieldProps) {
  const {
    name,
    label,
    description,
    required,
    defaultValue,
    value,
    onChange,
    def = {},
    templateAutocomplete,
    provider,
    dependsOnValues,
    placeholder,
    rows = 3,
    className,
    error,
    onResourceLabelChange,
    availableFileVariables = [],
  } = props;

  const fieldLabel = label ?? name;
  const type = typeof def.type === 'string' ? (def.type as string) : 'string';
  const enumOptions = Array.isArray(def.enum) ? def.enum.map(String) : undefined;
  const enumLabels = Array.isArray(def.enumLabels) ? (def.enumLabels as string[]) : undefined;
  const resourcePicker = def['x-resource-picker'] as ResourcePickerConfig | undefined;
  const htmlFor = `field-${name}`;
  const showError = Boolean(error);

  const baseProps = { id: htmlFor, placeholder, showError, templateAutocomplete };
  // File input fields have their own autocomplete using file variables, so they don't need templateAutocomplete
  const fileBaseProps = { id: htmlFor, placeholder, showError };

  // Helper: Render file input fields
  const renderFileInputField = () => {
    if (isFileInputArraySchema(def)) {
      return (
        <FileInputArrayField
          {...fileBaseProps}
          value={(value as FileInputValue[]) || []}
          onChange={onChange as (v: FileInputValue[]) => void}
          availableFileVariables={availableFileVariables}
        />
      );
    }
    if (isFileInputSchema(def)) {
      return (
        <FileInputField
          {...fileBaseProps}
          value={value as FileInputValue}
          onChange={onChange as (v: FileInputValue) => void}
          availableFileVariables={availableFileVariables}
        />
      );
    }
    return null;
  };

  // Helper: Render rich text field if applicable
  const renderRichTextField = () => {
    const uiType = def['x-ui-type'] as string | undefined;
    if (uiType !== 'rich_text') return null;

    const richTextOutput = def['x-rich-text-output'] as 'html' | 'markdown' | undefined;
    const richTextFeatures = def['x-rich-text-features'] as string[] | undefined;
    const maxLength = typeof def.maxLength === 'number' ? def.maxLength : undefined;

    return (
      <RichTextField
        id={htmlFor}
        value={String(value ?? '')}
        onChange={onChange as (v: string) => void}
        outputFormat={richTextOutput ?? 'html'}
        features={richTextFeatures}
        charLimit={maxLength}
        placeholder={placeholder}
        showError={showError}
        templateAutocomplete={templateAutocomplete}
        rows={rows}
      />
    );
  };

  // eslint-disable-next-line complexity
  const renderField = () => {
    // Check for file input fields first
    const fileField = renderFileInputField();
    if (fileField) return fileField;

    // Check for cron expression field (by field name)
    if (name === 'cron_expression' || name === 'cronExpression') {
      return (
        <CronScheduleField
          id={htmlFor}
          value={String(value ?? '')}
          onChange={onChange as (v: string) => void}
          placeholder={placeholder}
          showError={showError}
        />
      );
    }

    // Check for resource picker
    if (resourcePicker) {
      return (
        <ResourcePickerField
          {...baseProps}
          value={value}
          onChange={onChange}
          config={resourcePicker}
          provider={provider}
          dependsOnValues={dependsOnValues}
          fieldLabel={fieldLabel}
          fieldName={name}
          onResourceLabelChange={onResourceLabelChange}
          type={type}
        />
      );
    }

    // Check for enum field
    if (enumOptions?.length) {
      return <EnumSelectField {...baseProps} value={value} onChange={onChange as (v: string) => void} enumOptions={enumOptions} enumLabels={enumLabels} />;
    }

    // Type-specific fields
    if (type === 'boolean') {
      return <BooleanField {...baseProps} value={value} onChange={onChange as (v: boolean) => void} />;
    }

    if (type === 'integer' || type === 'number') {
      return (
        <NumericField
          {...baseProps}
          value={value}
          onChange={onChange}
          type={type}
          minimum={typeof def.minimum === 'number' ? def.minimum : undefined}
          maximum={typeof def.maximum === 'number' ? def.maximum : undefined}
        />
      );
    }

    if (type === 'array' || type === 'object') {
      return <JsonField {...baseProps} value={value} onChange={onChange} type={type} rows={rows} />;
    }

    // Check for rich text field
    const richText = renderRichTextField();
    if (richText) return richText;

    // Default: text field
    return <TextField {...baseProps} value={value} onChange={onChange as (v: string) => void} multiline={def.multiline === true} rows={rows} />;
  };

  return (
    <FormField
      label={fieldLabel}
      description={description || (def.description as string | undefined)}
      defaultValue={defaultValue ?? def.default}
      required={required}
      htmlFor={htmlFor}
      inputClassName={className}
    >
      {renderField()}
      {showError && <p className="text-xs text-destructive mt-1">{error}</p>}
    </FormField>
  );
}
