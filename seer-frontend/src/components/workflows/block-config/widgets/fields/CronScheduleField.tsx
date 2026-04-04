import { CronSchedulePicker } from './cron-schedule';

export interface CronScheduleFieldProps {
  id?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  showError?: boolean;
}

export function CronScheduleField({ value, onChange, showError }: CronScheduleFieldProps) {
  return (
    <CronSchedulePicker
      value={value}
      onChange={onChange}
      showError={showError}
    />
  );
}
