import { Checkbox } from '@/components/ui/checkbox';
import type { BooleanFieldProps } from './types';

export function BooleanField({ id, value, onChange }: BooleanFieldProps) {
  return (
    <div className="flex items-center gap-2">
      <Checkbox id={id} checked={Boolean(value)} onCheckedChange={checked => onChange(checked === true)} />
    </div>
  );
}
