/**
 * Types for TriggerEventPicker components
 */

import type { TriggerEventItem } from '@/types/triggers';

export interface TriggerEventPickerProps {
  /** Whether the picker dialog is open */
  open: boolean;
  /** Callback when open state changes */
  onOpenChange: (open: boolean) => void;
  /** The trigger instance ID */
  triggerId: string;
  /** The trigger key (e.g., 'poll.gmail.email_received') */
  triggerKey: string;
  /** Provider name (e.g., 'google', 'discord') */
  provider: string;
  /** Provider connection ID for polling triggers */
  providerConnectionId?: number;
  /** Subscription ID for persisted triggers */
  subscriptionId?: number;
  /** Filter params (e.g., channel_id for Slack/Discord) */
  filterParams?: Record<string, unknown>;
  /** Callback when an event is selected */
  onEventSelect: (event: TriggerEventItem) => void;
}

export interface TriggerEventListProps {
  items: TriggerEventItem[];
  selectedEventId?: string;
  onSelect: (event: TriggerEventItem) => void;
  hasNextPage: boolean;
  fetchNextPage: () => void;
  isFetchingNextPage: boolean;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
  triggerKey: string;
}

export interface TriggerEventListItemProps {
  item: TriggerEventItem;
  isSelected: boolean;
  onSelect: (item: TriggerEventItem) => void;
  triggerKey: string;
}

/** Mapping of trigger keys to their display metadata */
export interface TriggerDisplayConfig {
  icon: 'mail' | 'message-square' | 'zap' | 'database' | 'file-text' | 'user-check' | 'calendar' | 'clock' | 'sheet';
  label: string;
}

export const TRIGGER_DISPLAY_CONFIG: Record<string, TriggerDisplayConfig> = {
  'poll.gmail.email_received': {
    icon: 'mail',
    label: 'Gmail Email',
  },
  'poll.discord.message_received': {
    icon: 'message-square',
    label: 'Discord Message',
  },
  'poll.slack.message_received': {
    icon: 'message-square',
    label: 'Slack Message',
  },
  'poll.google_calendar.event_changed': {
    icon: 'calendar',
    label: 'Calendar Event',
  },
  'poll.google_calendar.event_start': {
    icon: 'clock',
    label: 'Calendar Event Start',
  },
  'poll.google_sheets.row_added': {
    icon: 'sheet',
    label: 'Google Sheets Row',
  },
  'webhook.generic': {
    icon: 'zap',
    label: 'Webhook Event',
  },
  'webhook.supabase.db_changes': {
    icon: 'database',
    label: 'Supabase Change',
  },
  'form.hosted': {
    icon: 'file-text',
    label: 'Form Submission',
  },
  'form.hitl': {
    icon: 'user-check',
    label: 'HITL Response',
  },
  'schedule.cron': {
    icon: 'clock',
    label: 'Scheduler',
  },
};

/** Browsing mode for trigger events */
export type TriggerBrowsingMode = 'polling' | 'persisted' | 'instant';

/** Mapping of trigger keys to their browsing mode */
export const TRIGGER_BROWSING_MODE: Record<string, TriggerBrowsingMode> = {
  'poll.gmail.email_received': 'polling',
  'poll.discord.message_received': 'polling',
  'poll.slack.message_received': 'polling',
  'poll.google_calendar.event_changed': 'polling',
  'poll.google_calendar.event_start': 'polling',
  'poll.google_sheets.row_added': 'polling',
  'webhook.generic': 'persisted',
  'webhook.supabase.db_changes': 'persisted',
  'form.hosted': 'persisted',
  'form.hitl': 'persisted',
  'schedule.cron': 'instant',
};
