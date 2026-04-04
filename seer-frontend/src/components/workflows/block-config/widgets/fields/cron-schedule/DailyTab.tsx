import { TimePicker } from './TimePicker';

interface DailyTabProps {
  hour: number;
  minute: number;
  onHourChange: (hour: number) => void;
  onMinuteChange: (minute: number) => void;
}

export function DailyTab({ hour, minute, onHourChange, onMinuteChange }: DailyTabProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-muted-foreground">Run every day</span>
      <TimePicker
        hour={hour}
        minute={minute}
        onHourChange={onHourChange}
        onMinuteChange={onMinuteChange}
      />
    </div>
  );
}
