import { ResourcePicker } from '@/components/workflows/dialogs/ResourcePicker';
import type { ResourcePickerFieldProps } from './types';

export function ResourcePickerField({
  id,
  value,
  onChange,
  config,
  provider,
  dependsOnValues,
  placeholder,
  fieldLabel,
  fieldName,
  onResourceLabelChange,
  type,
}: ResourcePickerFieldProps) {
  const current = value != null ? String(value) : undefined;

  const handleChange = (v: string, displayName?: string) => {
    const parsed =
      type === 'integer'
        ? (() => {
            const n = parseInt(v, 10);
            return Number.isNaN(n) ? v : n;
          })()
        : v;
    onChange(parsed);
    // Resource label bookkeeping is handled by section if needed
    if (onResourceLabelChange) {
      onResourceLabelChange(fieldName, displayName);
    }
  };

  return (
    <div className="space-y-1">
      <ResourcePicker
        config={config}
        provider={provider}
        value={current}
        onChange={handleChange}
        placeholder={placeholder || `Select ${fieldLabel}...`}
        dependsOnValues={dependsOnValues}
        className="text-xs"
      />
      {provider === 'supabase' && fieldName === 'integration_resource_id' && (
        <p className="text-[11px] text-muted-foreground mt-1">
          Connect Supabase Mgmt, then use "Bind project" to select a project.
        </p>
      )}
    </div>
  );
}
