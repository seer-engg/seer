/**
 * Dynamic form component that renders config inputs from a JSON Schema (config_schema).
 * Replaces hardcoded GmailConfigForm, CronConfigForm, SupabaseConfigForm.
 */

import { DynamicFormField } from '@/components/workflows/block-config/widgets/DynamicFormField';
import type { TemplateAutocompleteControls, ResourcePickerConfig } from '@/components/workflows/block-config/types';
import type { JsonObject } from '@/types/workflow-spec';

interface SchemaProperty {
  type?: string | string[];
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  pattern?: string;
  items?: JsonObject;
  'x-resource-picker'?: ResourcePickerConfig;
  [key: string]: unknown;
}

export interface DynamicTriggerConfigFormProps {
  configSchema: JsonObject | null;
  configValues: Record<string, unknown>;
  onConfigChange: (key: string, value: unknown) => void;
  provider?: string;
  templateAutocomplete: TemplateAutocompleteControls;
  validationErrors?: Record<string, string>;
  onResourceLabelChange?: (fieldName: string, label?: string) => void;
}

/**
 * Extracts field definitions from a config_schema.
 * Handles the JSON Schema structure where fields are defined in properties.
 */
function extractFieldsFromSchema(schema: JsonObject | null): Array<{
  name: string;
  def: SchemaProperty;
  required: boolean;
}> {
  if (!schema) {
    return [];
  }

  const properties = schema.properties as Record<string, SchemaProperty> | undefined;
  if (!properties || typeof properties !== 'object') {
    return [];
  }

  const requiredFields = Array.isArray(schema.required)
    ? (schema.required as string[])
    : [];

  return Object.entries(properties)
    .filter(([name]) => name !== 'provider_connection_id')
    .map(([name, def]) => ({
      name,
      def: def || {},
      required: requiredFields.includes(name),
    }));
}

/**
 * Converts snake_case to Title Case for display labels.
 */
function formatFieldLabel(name: string): string {
  return name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Determines a helpful placeholder based on the field definition.
 */
function getPlaceholder(name: string, def: SchemaProperty): string | undefined {
  if (def.default !== undefined) {
    return `Default: ${String(def.default)}`;
  }

  if (def.pattern) {
    return `Pattern: ${def.pattern}`;
  }

  if (def.minimum !== undefined && def.maximum !== undefined) {
    return `${def.minimum} - ${def.maximum}`;
  }

  return undefined;
}

export function DynamicTriggerConfigForm({
  configSchema,
  configValues,
  onConfigChange,
  provider,
  templateAutocomplete,
  validationErrors = {},
  onResourceLabelChange,
}: DynamicTriggerConfigFormProps) {
  const fields = extractFieldsFromSchema(configSchema);

  if (fields.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <div className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
        Configuration
      </div>
      {fields.map(({ name, def, required }) => {
        const value = configValues[name] ?? def.default ?? '';
        
        // Build dependsOnValues for resource pickers that have dependencies
        const resourcePicker = def['x-resource-picker'] as ResourcePickerConfig | undefined;
        const dependsOnKey = resourcePicker?.depends_on;
        const dependsOnValues = dependsOnKey 
          ? { [dependsOnKey]: configValues[dependsOnKey] as string } 
          : undefined;

        return (
          <DynamicFormField
            key={name}
            name={name}
            label={formatFieldLabel(name)}
            description={def.description}
            required={required}
            defaultValue={def.default}
            value={value}
            onChange={(v) => onConfigChange(name, v)}
            def={def}
            templateAutocomplete={templateAutocomplete}
            provider={provider}
            dependsOnValues={dependsOnValues}
            placeholder={getPlaceholder(name, def)}
            error={validationErrors[name]}
            onResourceLabelChange={onResourceLabelChange}
          />
        );
      })}
    </div>
  );
}
