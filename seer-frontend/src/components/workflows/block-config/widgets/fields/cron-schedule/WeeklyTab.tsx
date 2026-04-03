import { cn } from '@/lib/utils';
import { TimePicker } from './TimePicker';

interface WeeklyTabProps {
  days: number[];
  hour: number;
  minute: number;
  onDaysChange: (days: number[]) => void;
  onHourChange: (hour: number) => void;
  onMinuteChange: (minute: number) => void;
}

const DAYS = [
  { value: 0, label: 'Sun' },
  { value: 1, label: 'Mon' },
  { value: 2, label: 'Tue' },
  { value: 3, label: 'Wed' },
  { value: 4, label: 'Thu' },
  { value: 5, label: 'Fri' },
  { value: 6, label: 'Sat' },
];

export function WeeklyTab({ days, hour, minute, onDaysChange, onHourChange, onMinuteChange }: WeeklyTabProps) {
  const toggleDay = (day: number) => {
    if (days.includes(day)) {
      // Don't allow deselecting all days
      if (days.length > 1) {
        onDaysChange(days.filter(d => d !== day));
      }
    } else {
      onDaysChange([...days, day].sort((a, b) => a - b));
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {DAYS.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            onClick={() => toggleDay(value)}
            className={cn(
              'px-2.5 py-1 text-xs rounded-md border transition-colors',
              days.includes(value)
                ? 'bg-primary text-primary-foreground border-primary'
                : 'bg-background text-foreground border-input hover:bg-accent'
            )}
          >
            {label}
          </button>
        ))}
      </div>
      <TimePicker
        hour={hour}
        minute={minute}
        onHourChange={onHourChange}
        onMinuteChange={onMinuteChange}
      />
    </div>
  );
}
