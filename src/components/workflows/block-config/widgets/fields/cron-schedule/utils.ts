import type { CronScheduleState, ScheduleMode } from './types';
import { DEFAULT_CRON_STATE } from './types';

/**
 * Generate a 5-field cron expression from the schedule state
 * Format: minute hour day-of-month month day-of-week
 */
export function generateCronExpression(state: CronScheduleState): string {
  switch (state.mode) {
    case 'interval':
      return state.intervalUnit === 'minutes'
        ? `*/${state.intervalValue} * * * *`
        : `0 */${state.intervalValue} * * *`;
    case 'daily':
      return `${state.minute} ${state.hour} * * *`;
    case 'weekly': {
      const daysStr = state.days.length > 0 ? state.days.sort((a, b) => a - b).join(',') : '*';
      return `${state.minute} ${state.hour} * * ${daysStr}`;
    }
    case 'monthly':
      return `${state.minute} ${state.hour} ${state.dayOfMonth} * *`;
    case 'advanced':
      return state.rawExpression;
  }
}

// Helper to create state with common fields
function makeState(mode: ScheduleMode, raw: string, extra: Partial<CronScheduleState> = {}): CronScheduleState {
  return { ...DEFAULT_CRON_STATE, mode, rawExpression: raw, ...extra };
}

// Try to parse as interval pattern
function tryParseInterval(parts: string[], raw: string): CronScheduleState | null {
  const [minute, hour, dom, month, dow] = parts;
  if (dom !== '*' || month !== '*' || dow !== '*') return null;

  // */N * * * * (every N minutes)
  if (minute.startsWith('*/') && hour === '*') {
    const val = parseInt(minute.slice(2), 10);
    if (val > 0) return makeState('interval', raw, { intervalValue: val, intervalUnit: 'minutes' });
  }
  // 0 */N * * * (every N hours)
  if (minute === '0' && hour.startsWith('*/')) {
    const val = parseInt(hour.slice(2), 10);
    if (val > 0) return makeState('interval', raw, { intervalValue: val, intervalUnit: 'hours' });
  }
  return null;
}

// Try to parse as daily/weekly/monthly pattern
function tryParseScheduled(parts: string[], raw: string): CronScheduleState | null {
  const [minute, hour, dayOfMonth, month, dayOfWeek] = parts;
  const parsedMinute = parseInt(minute, 10);
  const parsedHour = parseInt(hour, 10);
  if (isNaN(parsedMinute) || isNaN(parsedHour)) return null;

  const timeFields = { hour: parsedHour, minute: parsedMinute };

  // Daily: N N * * *
  if (dayOfMonth === '*' && month === '*' && dayOfWeek === '*') {
    return makeState('daily', raw, timeFields);
  }
  // Weekly: N N * * <days>
  if (dayOfMonth === '*' && month === '*' && dayOfWeek !== '*') {
    const days = dayOfWeek.split(',').map(d => parseInt(d, 10)).filter(d => d >= 0 && d <= 6);
    if (days.length > 0) return makeState('weekly', raw, { ...timeFields, days });
  }
  // Monthly: N N <day> * *
  if (dayOfMonth !== '*' && month === '*' && dayOfWeek === '*') {
    const dom = parseInt(dayOfMonth, 10);
    if (dom >= 1 && dom <= 31) return makeState('monthly', raw, { ...timeFields, dayOfMonth: dom });
  }
  return null;
}

/**
 * Parse a cron expression back into schedule state for editing
 */
export function parseCronExpression(expression: string): CronScheduleState {
  const trimmed = expression.trim();
  if (!trimmed) return { ...DEFAULT_CRON_STATE };

  const parts = trimmed.split(/\s+/);
  if (parts.length !== 5) return makeState('advanced', trimmed);

  return tryParseInterval(parts, trimmed)
    ?? tryParseScheduled(parts, trimmed)
    ?? makeState('advanced', trimmed);
}

/**
 * Get human-readable description of a cron expression
 */
export function describeCronExpression(expression: string): string {
  const state = parseCronExpression(expression);

  switch (state.mode) {
    case 'interval':
      return state.intervalUnit === 'minutes'
        ? `Every ${state.intervalValue} minute${state.intervalValue > 1 ? 's' : ''}`
        : `Every ${state.intervalValue} hour${state.intervalValue > 1 ? 's' : ''}`;

    case 'daily':
      return `Daily at ${formatTime(state.hour, state.minute)}`;

    case 'weekly': {
      const dayNames = state.days.map(d => DAY_NAMES[d]).join(', ');
      return `${dayNames} at ${formatTime(state.hour, state.minute)}`;
    }

    case 'monthly':
      return `Day ${state.dayOfMonth} of every month at ${formatTime(state.hour, state.minute)}`;

    case 'advanced':
      return expression;
  }
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function formatTime(hour: number, minute: number): string {
  const h = hour % 12 || 12;
  const ampm = hour >= 12 ? 'PM' : 'AM';
  const m = minute.toString().padStart(2, '0');
  return `${h}:${m} ${ampm}`;
}

/**
 * Detect which mode a cron expression belongs to
 */
export function detectMode(expression: string): ScheduleMode {
  return parseCronExpression(expression).mode;
}

/**
 * Validate a cron expression (basic validation)
 */
export function isValidCronExpression(expression: string): boolean {
  const trimmed = expression.trim();
  if (!trimmed) return false;

  const parts = trimmed.split(/\s+/);
  if (parts.length !== 5) return false;

  // Basic field validation regex
  const fieldRegex = /^(\*|[0-9]+(-[0-9]+)?(\/[0-9]+)?)(,(\*|[0-9]+(-[0-9]+)?(\/[0-9]+)?))*$/;
  return parts.every(part => fieldRegex.test(part));
}
