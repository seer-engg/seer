import type { JsonObject, TriggerSpec } from '@/types/workflow-spec';

export interface TriggerDescriptor {
  key: string;
  title: string;
  provider: string;
  mode: string;
  description?: string | null;
  event_schema: JsonObject;
  filter_schema?: JsonObject | null;
  config_schema?: JsonObject | null;
  sample_event?: JsonObject | null;
  metadata?: Record<string, unknown> | null;
  requires_connection?: boolean;
  is_connected?: boolean;  // Backend now returns connection status
}

// Type alias for consistency across the codebase
export type TriggerCatalogEntry = TriggerDescriptor;

export interface TriggerCatalogResponse {
  triggers: TriggerDescriptor[];
}

export type WorkflowTrigger = TriggerSpec;

export interface TriggerSubscriptionCreateRequest extends TriggerSpec {
  workflow_id: string;
  trigger_key: string;
}

export type TriggerSubscriptionUpdateRequest = Partial<TriggerSpec>;

export interface TriggerSubscriptionResponse extends TriggerSpec {
  workflow_id?: string;
}

export interface TriggerSubscriptionListResponse {
  items: TriggerSubscriptionResponse[];
}

export interface TriggerSubscriptionTestRequest {
  event?: JsonObject;
}

export interface TriggerSubscriptionTestResponse {
  inputs: JsonObject;
  errors: string[];
}

/**
 * A single trigger event item returned from the browse endpoint.
 * Used for displaying real events (emails, messages) in the TriggerEventPicker.
 */
export interface TriggerEventItem {
  /** Unique identifier for this event (e.g., message ID) */
  id: string;
  /** Primary display text (e.g., email subject + sender) */
  display_title: string;
  /** Secondary display text (e.g., timestamp) */
  display_subtitle?: string;
  /** Preview/snippet of the content */
  preview?: string;
  /** Full trigger event data to pass when running workflow */
  envelope: Record<string, unknown>;
}

/**
 * Response from the trigger events browse endpoint.
 * Supports pagination via next_page_token.
 */
export interface TriggerEventsResponse {
  items: TriggerEventItem[];
  next_page_token?: string;
}

/**
 * Extended subscription info for management list view.
 */
export interface TriggerSubscriptionListItem {
  id: number;
  trigger_id: string;
  trigger_key: string;
  title: string | null;
  enabled: boolean;
  workflow_id: string;
  workflow_title: string;
  last_event_at: string | null;
  created_at: string;
}

/**
 * Response for trigger subscription list with extended info.
 */
export interface TriggerSubscriptionListItemsResponse {
  items: TriggerSubscriptionListItem[];
}

/**
 * Filters for listing trigger subscriptions.
 */
export interface TriggerSubscriptionFilters {
  workflow_id?: string;
  trigger_key?: string;
  search?: string;
}
