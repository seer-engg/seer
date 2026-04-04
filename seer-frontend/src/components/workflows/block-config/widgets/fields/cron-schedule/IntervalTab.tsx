import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import type { IntervalUnit } from './types';

interface IntervalTabProps {
  value: number;
  unit: IntervalUnit;
  onValueChange: (value: number) => void;
  onUnitChange: (unit: IntervalUnit) => void;
}

export function IntervalTab({ value, unit, onValueChange, onUnitChange }: IntervalTabProps) {
  const handleValueChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const num = parseInt(e.target.value, 10);
    if (!isNaN(num) && num >= 1 && num <= 60) {
      onValueChange(num);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Run every</span>
      <Input
        type="number"
        min={1}
        max={60}
        value={value}
        onChange={handleValueChange}
        className="w-[60px] text-xs h-8"
      />
      <Select value={unit} onValueChange={(v) => onUnitChange(v as IntervalUnit)}>
        <SelectTrigger className="w-[90px] text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="minutes" className="text-xs">minutes</SelectItem>
          <SelectItem value="hours" className="text-xs">hours</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}
