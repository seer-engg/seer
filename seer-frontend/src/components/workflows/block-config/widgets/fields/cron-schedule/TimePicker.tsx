import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface TimePickerProps {
  hour: number;
  minute: number;
  onHourChange: (hour: number) => void;
  onMinuteChange: (minute: number) => void;
}

// Generate hour options (0-23)
const HOURS = Array.from({ length: 24 }, (_, i) => i);
// Generate minute options (0, 5, 10, ... 55) - 5-minute increments for cleaner UX
const MINUTES = Array.from({ length: 12 }, (_, i) => i * 5);

export function TimePicker({ hour, minute, onHourChange, onMinuteChange }: TimePickerProps) {
  // Round minute to nearest 5 for display
  const displayMinute = Math.round(minute / 5) * 5;

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">at</span>
      <Select value={String(hour)} onValueChange={(v) => onHourChange(parseInt(v, 10))}>
        <SelectTrigger className="w-[70px] text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {HOURS.map((h) => (
            <SelectItem key={h} value={String(h)} className="text-xs">
              {formatHour(h)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">:</span>
      <Select value={String(displayMinute)} onValueChange={(v) => onMinuteChange(parseInt(v, 10))}>
        <SelectTrigger className="w-[60px] text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {MINUTES.map((m) => (
            <SelectItem key={m} value={String(m)} className="text-xs">
              {m.toString().padStart(2, '0')}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">{hour >= 12 ? 'PM' : 'AM'}</span>
    </div>
  );
}

function formatHour(hour: number): string {
  if (hour === 0) return '12';
  if (hour <= 12) return String(hour);
  return String(hour - 12);
}
