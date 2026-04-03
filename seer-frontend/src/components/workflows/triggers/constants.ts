import type { LucideIcon } from 'lucide-react';
import { Calendar, CalendarClock, CalendarDays, Database, Link, Mail, FormInput, MessageSquare, Hash, Table2, Sheet, Video } from 'lucide-react';
import { SHOW_MEETINGS } from '@/lib/feature-flags';

export const WEBHOOK_TRIGGER_KEY = 'webhook.generic';
export const GMAIL_TRIGGER_KEY = 'poll.gmail.email_received';
export const CRON_TRIGGER_KEY = 'schedule.cron';
export const SUPABASE_TRIGGER_KEY = 'webhook.supabase.db_changes';
export const FORM_TRIGGER_KEY = 'form.hosted';
export const DISCORD_TRIGGER_KEY = 'poll.discord.message_received';
export const SLACK_TRIGGER_KEY = 'poll.slack.message_received';
export const CALENDAR_TRIGGER_KEY = 'poll.google_calendar.event_changed';
export const CALENDAR_START_TRIGGER_KEY = 'poll.google_calendar.event_start';
export const AIRTABLE_TRIGGER_KEY = 'poll.airtable.new_record_in_view';
export const GOOGLE_SHEETS_TRIGGER_KEY = 'poll.google_sheets.row_added';
export const MEETINGS_EA_TRIGGER_KEY = 'webhook.meetings_ea.bot_event';

export interface GmailFieldOption {
  label: string;
  path: string;
  hint?: string;
}

export const GMAIL_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Subject', path: 'data.subject' },
  { label: 'Snippet', path: 'data.snippet' },
  { label: 'Message ID', path: 'data.message_id' },
  { label: 'Thread ID', path: 'data.thread_id' },
  { label: 'From email', path: 'data.from.email' },
  { label: 'From name', path: 'data.from.name' },
  { label: 'First recipient email', path: 'data.to.0.email' },
  { label: 'Internal date (ms)', path: 'data.internal_date_ms' },
];

export const CRON_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Scheduled time', path: 'data.scheduled_time' },
  { label: 'Actual time', path: 'data.actual_time' },
  { label: 'Cron expression', path: 'data.cron_expression' },
  { label: 'Timezone', path: 'data.timezone' },
];

export const DISCORD_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Message ID', path: 'data.message_id' },
  { label: 'Channel ID', path: 'data.channel_id' },
  { label: 'Guild ID', path: 'data.guild_id' },
  { label: 'Author ID', path: 'data.author.id' },
  { label: 'Author username', path: 'data.author.username' },
  { label: 'Content', path: 'data.content' },
  { label: 'Timestamp', path: 'data.timestamp' },
  { label: 'Mentions bot', path: 'data.mentions_bot' },
];

export const SLACK_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Message ID', path: 'data.message_id' },
  { label: 'Channel ID', path: 'data.channel_id' },
  { label: 'Workspace ID', path: 'data.workspace_id' },
  { label: 'User ID', path: 'data.user.id' },
  { label: 'Username', path: 'data.user.username' },
  { label: 'Text', path: 'data.text' },
  { label: 'Timestamp', path: 'data.timestamp' },
  { label: 'Mentions Bot', path: 'data.mentions_bot' },
];

export const CALENDAR_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Event ID', path: 'data.event_id' },
  { label: 'Event type', path: 'data.event_type' },
  { label: 'Summary', path: 'data.summary' },
  { label: 'Description', path: 'data.description' },
  { label: 'Location', path: 'data.location' },
  { label: 'Start time', path: 'data.start.datetime' },
  { label: 'End time', path: 'data.end.datetime' },
  { label: 'Organizer email', path: 'data.organizer.email' },
  { label: 'Is all day', path: 'data.is_all_day' },
  { label: 'Conference link', path: 'data.conference_link', hint: 'Google Meet / Zoom URL' },
  { label: 'Calendar ID', path: 'data.calendar_id' },
  { label: 'HTML link', path: 'data.html_link' },
];

export const CALENDAR_START_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Event ID', path: 'data.event_id' },
  { label: 'Trigger type', path: 'data.trigger_type' },
  { label: 'Offset minutes', path: 'data.offset_minutes' },
  { label: 'Trigger time', path: 'data.trigger_time' },
  { label: 'Event start', path: 'data.event_start' },
  { label: 'Summary', path: 'data.summary' },
  { label: 'Description', path: 'data.description' },
  { label: 'Location', path: 'data.location' },
  { label: 'Start time', path: 'data.start.datetime' },
  { label: 'End time', path: 'data.end.datetime' },
  { label: 'Organizer email', path: 'data.organizer.email' },
  { label: 'Is all day', path: 'data.is_all_day' },
  { label: 'Conference link', path: 'data.conference_link', hint: 'Google Meet / Zoom URL' },
  { label: 'Calendar ID', path: 'data.calendar_id' },
  { label: 'HTML link', path: 'data.html_link' },
];

export const MEETINGS_EA_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Event type', path: 'data.event', hint: 'e.g. bot.state_change, transcript.update' },
  { label: 'Bot ID', path: 'data.data.id' },
  { label: 'Bot state', path: 'data.data.state', hint: 'e.g. joining, joined_recording, ended' },
  { label: 'Meeting URL', path: 'data.data.meeting_url' },
];

export const AIRTABLE_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Record ID', path: 'data.record_id' },
  { label: 'Created time', path: 'data.created_time' },
  { label: 'Fields', path: 'data.fields' },
  { label: 'Base ID', path: 'data.base_id' },
  { label: 'Table ID', path: 'data.table_id' },
  { label: 'View ID', path: 'data.view_id' },
];

export const GOOGLE_SHEETS_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Row number', path: 'data.row_number' },
  { label: 'Fields', path: 'data.fields' },
  { label: 'Row values', path: 'data.row_values' },
  { label: 'Spreadsheet ID', path: 'data.spreadsheet_id' },
  { label: 'Sheet name', path: 'data.sheet_name' },
];

export const TRIGGER_ICON_BY_KEY: Record<string, LucideIcon> = {
  [WEBHOOK_TRIGGER_KEY]: Link,
  [GMAIL_TRIGGER_KEY]: Mail,
  [CRON_TRIGGER_KEY]: Calendar,
  [SUPABASE_TRIGGER_KEY]: Database,
  [FORM_TRIGGER_KEY]: FormInput,
  [DISCORD_TRIGGER_KEY]: MessageSquare,
  [SLACK_TRIGGER_KEY]: Hash,
  [CALENDAR_TRIGGER_KEY]: CalendarDays,
  [CALENDAR_START_TRIGGER_KEY]: CalendarClock,
  [AIRTABLE_TRIGGER_KEY]: Table2,
  [GOOGLE_SHEETS_TRIGGER_KEY]: Sheet,
  ...(SHOW_MEETINGS ? { [MEETINGS_EA_TRIGGER_KEY]: Video } : {}),
};

// Trigger colors for build panel icons
export const TRIGGER_COLOR_BY_KEY: Record<string, { bg: string; text: string }> = {
  [WEBHOOK_TRIGGER_KEY]: { bg: 'bg-slate-500/10', text: 'text-slate-500' },
  [GMAIL_TRIGGER_KEY]: { bg: 'bg-red-500/10', text: 'text-red-500' },
  [CRON_TRIGGER_KEY]: { bg: 'bg-purple-500/10', text: 'text-purple-500' },
  [SUPABASE_TRIGGER_KEY]: { bg: 'bg-emerald-500/10', text: 'text-emerald-500' },
  [FORM_TRIGGER_KEY]: { bg: 'bg-blue-500/10', text: 'text-blue-500' },
  [DISCORD_TRIGGER_KEY]: { bg: 'bg-indigo-500/10', text: 'text-indigo-500' },
  [SLACK_TRIGGER_KEY]: { bg: 'bg-fuchsia-500/10', text: 'text-fuchsia-500' },
  [CALENDAR_TRIGGER_KEY]: { bg: 'bg-sky-500/10', text: 'text-sky-500' },
  [CALENDAR_START_TRIGGER_KEY]: { bg: 'bg-sky-500/10', text: 'text-sky-500' },
  [AIRTABLE_TRIGGER_KEY]: { bg: 'bg-yellow-500/10', text: 'text-yellow-500' },
  [GOOGLE_SHEETS_TRIGGER_KEY]: { bg: 'bg-green-500/10', text: 'text-green-500' },
  ...(SHOW_MEETINGS ? { [MEETINGS_EA_TRIGGER_KEY]: { bg: 'bg-violet-500/10', text: 'text-violet-500' } } : {}),
};

export const CRON_PRESETS = [
  { label: 'Every 5 minutes', expression: '*/5 * * * *' },
  { label: 'Every 15 minutes', expression: '*/15 * * * *' },
  { label: 'Every 30 minutes', expression: '*/30 * * * *' },
  { label: 'Every hour', expression: '0 * * * *' },
  { label: 'Every day at midnight', expression: '0 0 * * *' },
  { label: 'Every day at 9 AM', expression: '0 9 * * *' },
  { label: 'Every Monday at 9 AM', expression: '0 9 * * 1' },
  { label: 'First day of month', expression: '0 0 1 * *' },
];

export const CALENDAR_OFFSET_PRESETS = [
  { label: '5 minutes before', value: -5 },
  { label: '15 minutes before', value: -15 },
  { label: '30 minutes before', value: -30 },
  { label: '1 hour before', value: -60 },
  { label: 'At event start', value: 0 },
  { label: '5 minutes after', value: 5 },
  { label: '15 minutes after', value: 15 },
];

export const TIMEZONE_OPTIONS = [
  'UTC',
  'America/New_York',
  'America/Chicago',
  'America/Denver',
  'America/Los_Angeles',
  'America/Anchorage',
  'Pacific/Honolulu',
  'America/Phoenix',
  'America/Toronto',
  'America/Vancouver',
  'America/Mexico_City',
  'America/Sao_Paulo',
  'America/Buenos_Aires',
  'Europe/London',
  'Europe/Paris',
  'Europe/Berlin',
  'Europe/Amsterdam',
  'Europe/Rome',
  'Europe/Madrid',
  'Europe/Moscow',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Asia/Hong_Kong',
  'Asia/Singapore',
  'Asia/Kolkata',
  'Asia/Dubai',
  'Asia/Seoul',
  'Asia/Bangkok',
  'Asia/Jakarta',
  'Asia/Karachi',
  'Asia/Dhaka',
  'Australia/Sydney',
  'Australia/Melbourne',
  'Australia/Perth',
  'Pacific/Auckland',
  'Africa/Cairo',
  'Africa/Lagos',
  'Africa/Johannesburg',
  'Africa/Nairobi',
];

export const SUPABASE_FIELD_OPTIONS: GmailFieldOption[] = [
  { label: 'Change type', path: 'data.type' },
  { label: 'Schema', path: 'data.schema' },
  { label: 'Table', path: 'data.table' },
  { label: 'New record', path: 'data.record' },
  { label: 'Old record', path: 'data.old_record' },
];

export const GMAIL_TOOL_FALLBACK_NAMES = ['gmail_read_emails'];
export const DISCORD_TOOL_FALLBACK_NAMES = ['discord_send_message'];
export const SUPABASE_TOOL_FALLBACK_NAMES = ['supabase_table_query'];
export const SLACK_TOOL_FALLBACK_NAMES = ['slack_send_message'];
export const CALENDAR_TOOL_FALLBACK_NAMES = ['google_calendar_list_events'];
export const CALENDAR_START_TOOL_FALLBACK_NAMES = ['google_calendar_list_events'];
export const AIRTABLE_TOOL_FALLBACK_NAMES = ['airtable_list_records', 'airtable_create_record'];
export const GOOGLE_SHEETS_TOOL_FALLBACK_NAMES = ['google_sheets_read', 'google_sheets_batch_read'];
export const MEETINGS_EA_TOOL_FALLBACK_NAMES = ['meetings_ea_get_transcript', 'meetings_ea_get_recording'];

