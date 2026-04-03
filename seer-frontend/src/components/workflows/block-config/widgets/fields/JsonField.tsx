import { cn } from '@/lib/utils';
import { AutocompleteTextarea } from '../AutocompleteTextarea';
import { safeJSONParse, safeJSONStringify } from '../../utils/param-utils';
import type { JsonFieldProps } from './types';

export function JsonField({
  id,
  value,
  onChange,
  placeholder,
  templateAutocomplete,
  showError,
  type,
  rows = 3,
}: JsonFieldProps) {
  const stringValue = safeJSONStringify(value);
  const defaultPlaceholder = type === 'array' ? '[]' : '{}';

  return (
    <AutocompleteTextarea
      id={id}
      value={stringValue}
      onChange={v => onChange(safeJSONParse(v))}
      placeholder={placeholder || defaultPlaceholder}
      templateAutocomplete={templateAutocomplete}
      rows={rows}
      className={cn('text-xs font-mono', showError && 'border-destructive')}
    />
  );
}
