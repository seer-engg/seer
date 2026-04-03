export type ScheduleMode = 'interval' | 'daily' | 'weekly' | 'monthly' | 'advanced';

export type IntervalUnit = 'minutes' | 'hours';

export interface CronScheduleState {
  mode: ScheduleMode;
  // Interval mode
  intervalValue: number;
  intervalUnit: IntervalUnit;
  // Daily/Weekly/Monthly shared
  hour: number;
  minute: number;
  // Weekly specific
  days: number[]; // 0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
  // Monthly specific
  dayOfMonth: number;
  // Advanced mode
  rawExpression: string;
}

export const DEFAULT_CRON_STATE: CronScheduleState = {
  mode: 'interval',
  intervalValue: 5,
  intervalUnit: 'minutes',
  hour: 9,
  minute: 0,
  days: [1], // Monday
  dayOfMonth: 1,
  rawExpression: '*/5 * * * *',
};

export interface CronPreset {
  label: string;
  expression: string;
}
