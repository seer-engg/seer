import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { TimePicker } from './TimePicker';

interface MonthlyTabProps {
  dayOfMonth: number;
  hour: number;
  minute: number;
  onDayOfMonthChange: (day: number) => void;
  onHourChange: (hour: number) => void;
  onMinuteChange: (minute: number) => void;
}

// Days 1-31
const DAYS_OF_MONTH = Array.from({ length: 31 }, (_, i) => i + 1);

export function MonthlyTab({
  dayOfMonth,
  hour,
  minute,
  onDayOfMonthChange,
  onHourChange,
  onMinuteChange,
}: MonthlyTabProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">On day</span>
      <Select value={String(dayOfMonth)} onValueChange={(v) => onDayOfMonthChange(parseInt(v, 10))}>
        <SelectTrigger className="w-[60px] text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {DAYS_OF_MONTH.map((d) => (
            <SelectItem key={d} value={String(d)} className="text-xs">
              {d}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">of every month</span>
      <TimePicker
        hour={hour}
        minute={minute}
        onHourChange={onHourChange}
        onMinuteChange={onMinuteChange}
      />
    </div>
  );
}
