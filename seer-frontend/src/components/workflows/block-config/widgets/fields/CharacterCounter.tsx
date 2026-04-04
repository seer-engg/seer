import { cn } from '@/lib/utils';

interface CharacterCounterProps {
  charCount: number;
  charLimit: number;
}

export function CharacterCounter({ charCount, charLimit }: CharacterCounterProps) {
  return (
    <div
      className={cn(
        'border-t border-border px-3 py-1 text-xs text-muted-foreground',
        charCount > charLimit * 0.9 && 'text-warning',
        charCount >= charLimit && 'text-destructive'
      )}
    >
      {charCount.toLocaleString()} / {charLimit.toLocaleString()} characters
    </div>
  );
}
