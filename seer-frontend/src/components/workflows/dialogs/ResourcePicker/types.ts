/**
 * Shared types for ResourcePicker components
 */

export type JsonRecord = Record<string, unknown>;

export interface ResourcePickerConfig {
  resource_type: string;
  display_field?: string;
  value_field?: string;
  search_enabled?: boolean;
  hierarchy?: boolean;
  filter?: Record<string, unknown>;
  depends_on?: string;
  endpoint?: string;
}

export interface ResourceItem {
  id: string;
  name: string;
  display_name: string;
  type?: string;
  mime_type?: string;
  icon_url?: string;
  web_url?: string;
  modified_time?: string;
  has_children?: boolean;
  description?: string;
  raw?: JsonRecord;
  project_ref?: string;
  project_id?: string | number;
  resource_key?: string;
  region?: string;
}

export interface ResourceBrowseResponse {
  items: ResourceItem[];
  next_page_token?: string;
  supports_hierarchy?: boolean;
  supports_search?: boolean;
  error?: string;
}

export interface ResourcePickerProps {
  /** Resource picker configuration from tool schema */
  config: ResourcePickerConfig;
  /** OAuth provider (google, github). Optional if config.endpoint is provided */
  provider?: string;
  /** Current selected value (resource ID) */
  value?: string;
  /** Callback when resource is selected */
  onChange: (value: string, displayName?: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Whether the picker is disabled */
  disabled?: boolean;
  /** Dependent parameter values */
  dependsOnValues?: Record<string, string>;
  /** Custom trigger button class */
  className?: string;
}

