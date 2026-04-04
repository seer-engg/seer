export const GMAIL_QUICK_OPTIONS = [
  { label: 'Subject', path: 'data.subject' },
  { label: 'From', path: 'data.from' },
  { label: 'Body', path: 'data.body' },
  { label: 'Message ID', path: 'data.messageId' },
];

export const CRON_QUICK_OPTIONS = [
  { label: 'Timestamp', path: 'data.timestamp' },
  { label: 'Run ID', path: 'data.run_id' },
];

export const SUPABASE_QUICK_OPTIONS = [
  { label: 'Record', path: 'data.record' },
  { label: 'Old Record', path: 'data.old_record' },
  { label: 'Event Type', path: 'data.type' },
  { label: 'Table', path: 'data.table' },
];

export const SLACK_QUICK_OPTIONS = [
  { label: 'Text', path: 'data.text' },
  { label: 'Channel ID', path: 'data.channel_id' },
  { label: 'User', path: 'data.user.username' },
  { label: 'Timestamp', path: 'data.timestamp' },
];

export type TriggerKind = 'gmail' | 'cron' | 'supabase' | 'webhook' | 'discord' | 'form' | 'slack' | 'calendar' | 'airtable' | 'google_sheets';

export const DISCORD_QUICK_OPTIONS = [
  { label: 'Content', path: 'data.content' },
  { label: 'Channel ID', path: 'data.channel_id' },
  { label: 'Author', path: 'data.author.username' },
  { label: 'Timestamp', path: 'data.timestamp' },
];

export const FORM_QUICK_OPTIONS = [
  { label: 'Form Data', path: 'data' },
];

export const CALENDAR_QUICK_OPTIONS = [
  { label: 'Summary', path: 'data.summary' },
  { label: 'Event ID', path: 'data.event_id' },
  { label: 'Event Type', path: 'data.event_type' },
  { label: 'Start Time', path: 'data.start.datetime' },
];

export const AIRTABLE_QUICK_OPTIONS = [
  { label: 'Record ID', path: 'data.record_id' },
  { label: 'Fields', path: 'data.fields' },
  { label: 'Created Time', path: 'data.created_time' },
  { label: 'Base ID', path: 'data.base_id' },
];

export const GOOGLE_SHEETS_QUICK_OPTIONS = [
  { label: 'Row Number', path: 'data.row_number' },
  { label: 'Fields', path: 'data.fields' },
  { label: 'Spreadsheet ID', path: 'data.spreadsheet_id' },
  { label: 'Sheet Name', path: 'data.sheet_name' },
];

export const QUICK_OPTIONS_BY_KIND: Record<Exclude<TriggerKind, 'webhook'>, { label: string; path: string }[]> = {
  gmail: GMAIL_QUICK_OPTIONS,
  cron: CRON_QUICK_OPTIONS,
  supabase: SUPABASE_QUICK_OPTIONS,
  slack: SLACK_QUICK_OPTIONS,
  discord: DISCORD_QUICK_OPTIONS,
  form: FORM_QUICK_OPTIONS,
  calendar: CALENDAR_QUICK_OPTIONS,
  airtable: AIRTABLE_QUICK_OPTIONS,
  google_sheets: GOOGLE_SHEETS_QUICK_OPTIONS,
};
