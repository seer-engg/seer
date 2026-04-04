import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Wand2 } from 'lucide-react';
import { CRON_PRESETS } from '@/components/workflows/triggers/constants';

interface PresetsDropdownProps {
  currentExpression: string;
  onSelect: (expression: string) => void;
}

export function PresetsDropdown({ currentExpression, onSelect }: PresetsDropdownProps) {
  const currentPreset = CRON_PRESETS.find(p => p.expression === currentExpression);

  return (
    <div className="flex items-center gap-2">
      <Wand2 className="h-3.5 w-3.5 text-muted-foreground" />
      <Select value={currentPreset?.expression ?? 'custom'} onValueChange={onSelect}>
        <SelectTrigger className="flex-1 text-xs h-8">
          <SelectValue placeholder="Quick select..." />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="custom" className="text-xs text-muted-foreground">
            Custom schedule
          </SelectItem>
          {CRON_PRESETS.map((preset) => (
            <SelectItem key={preset.expression} value={preset.expression} className="text-xs">
              {preset.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
