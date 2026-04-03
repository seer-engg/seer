import { cn } from '@/lib/utils';
import { AutocompleteInput } from '../AutocompleteInput';
import { AutocompleteTextarea } from '../AutocompleteTextarea';
import type { TextFieldProps } from './types';

export function TextField({
  id,
  value,
  onChange,
  placeholder,
  templateAutocomplete,
  showError,
  multiline,
  rows = 3,
}: TextFieldProps) {
  if (multiline) {
    // Calculate max-height based on rows (approx 24px per row + padding)
    const maxHeight = Math.max(120, rows * 24 + 16);

    return (
      <AutocompleteTextarea
        id={id}
        value={String(value ?? '')}
        onChange={onChange}
        placeholder={placeholder}
        templateAutocomplete={templateAutocomplete}
        rows={rows}
        className={cn('overflow-y-auto', showError && 'border-destructive')}
        style={{ maxHeight: `${maxHeight}px` }}
      />
    );
  }

  return (
    <AutocompleteInput
      id={id}
      value={String(value ?? '')}
      onChange={onChange}
      placeholder={placeholder}
      templateAutocomplete={templateAutocomplete}
      className={cn('text-xs', showError && 'border-destructive')}
    />
  );
}
