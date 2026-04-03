import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { isValidCronExpression } from './utils';

interface AdvancedInputProps {
  value: string;
  onChange: (value: string) => void;
}

export function AdvancedInput({ value, onChange }: AdvancedInputProps) {
  return (
    <div className="space-y-2">
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="*/5 * * * *"
        className={cn(
          'font-mono text-xs h-16 resize-none',
          value && !isValidCronExpression(value) && 'border-destructive'
        )}
      />
      <p className="text-xs text-muted-foreground">
        Format: minute hour day-of-month month day-of-week
      </p>
    </div>
  );
}
