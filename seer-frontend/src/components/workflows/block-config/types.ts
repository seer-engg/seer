import type { Dispatch, SetStateAction } from 'react';
import { useTemplateAutocomplete } from '../../../hooks/useTemplateAutocomplete';

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

export interface ToolMetadata {
  name: string;
  description: string;
  provider?: string | null;
  integration_type?: string | null;
  output_schema?: Record<string, unknown> | null;
  parameters?: {
    properties?: Record<string, unknown>;
    required?: string[];
  };
}

export type TemplateAutocompleteControls = ReturnType<typeof useTemplateAutocomplete>;

/**
 * Base interface for all block configurations
 */
export interface BaseBlockConfig {
  [key: string]: unknown;
}

/**
 * For Loop Block Configuration
 */
export interface ForLoopBlockConfig extends BaseBlockConfig {
  array_variable?: string;
  array_var?: string; // Legacy support
  item_var?: string;
  item_variable?: string;
  index_var?: string;
  // Legacy fields (read-only)
  array_literal?: unknown[];
  array_mode?: string;
}

/**
 * If/Else Block Configuration
 */
export interface IfElseBlockConfig extends BaseBlockConfig {
  condition?: string;
  condition_type?: 'expression' | 'comparison';
}

/**
 * Tool Block Configuration
 */
export interface ToolBlockConfig extends BaseBlockConfig {
  tool_name?: string;
  toolName?: string; // Legacy support
  connection_id?: number | null;  // Database ID of OAuth connection for multi-account support
  provider?: string;
  integration_type?: string;  // Integration type (e.g., 'gmail', 'google_sheets')
  arguments?: Record<string, unknown>;
  params?: Record<string, unknown>;
}

/**
 * MCP Block Configuration
 */
export interface McpToolInfo {
  name: string;
  description: string;
  input_schema?: Record<string, unknown>;
}

export interface McpBlockConfig extends BaseBlockConfig {
  server?: string;
  server_type?: 'http' | 'stdio';
  tool?: string;
  inputs?: Record<string, unknown>;
  auth?: Record<string, unknown>;
  mcp_server_config_id?: number | null;
  fetchedTools?: McpToolInfo[];
  // Structured editing fields (transient UI state, serialized to server/auth for backend)
  stdio_command?: string;
  stdio_args?: string[];
  stdio_env?: Record<string, string>;
  http_headers?: Record<string, string>;
}

export interface SavedMcpServer {
  id: number;
  name: string;
  server_url: string;
  server_type: 'http' | 'stdio';
  has_auth: boolean;
  created_at?: string | null;
  updated_at?: string | null;
}

/**
 * Code Block Configuration
 */
export interface CodeBlockConfig extends BaseBlockConfig {
  code?: string;
  language?: 'python' | 'javascript' | 'typescript';
}

/**
 * Input Block Configuration
 */
export interface InputBlockConfig extends BaseBlockConfig {
  fields?: Array<{
    id: string;
    label: string;
    type: 'text' | 'number' | 'email' | 'textarea' | 'select' | 'checkbox';
    required?: boolean;
    placeholder?: string;
    options?: string[];
    default_value?: string | number | boolean;
  }>;
}

/**
 * HITL (Human-in-the-Loop) Block Configuration
 */
export interface HitlInputOption {
  value: string;
  label: string;
  requires_text?: boolean;
}

export interface HitlInputField {
  id: string;
  question: string;
  input_type: 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean' | 'table';
  options?: HitlInputOption[];
  required?: boolean;
  placeholder?: string;
  default_value?: string | number | boolean;
  columns?: HitlTableColumn[];
  row_data_expression?: string;
  row_display_fields?: HitlDisplayItem[];
}

export interface HitlTableColumn {
  id: string;
  header: string;
  input_type: 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean';
  options?: HitlInputOption[];
  required?: boolean;
}

export interface HitlDisplayItem {
  label: string;
  value: string; // Template expression like ${node.field}
}

/**
 * Delivery channel types for HITL notifications
 */
export type DeliveryChannelType = 'platform' | 'gmail';

export interface GmailDeliveryConfig {
  /** Email address to send HITL form link to. Supports templates like ${trigger.sender_email} */
  recipient_email: string;
}

export interface HITLDeliveryChannel {
  type: DeliveryChannelType;
  gmail?: GmailDeliveryConfig;
}

export interface HitlBlockConfig extends BaseBlockConfig {
  title?: string;
  description?: string;
  display?: HitlDisplayItem[];
  inputs?: HitlInputField[];
  timeout_seconds?: number;
  output_schema?: Record<string, unknown>;
  /** Channels to deliver HITL notifications. Default: [{ type: 'platform' }] */
  delivery_channels?: HITLDeliveryChannel[];
}

/**
 * Browser Block Configuration
 */
export interface BrowserBlockConfig extends BaseBlockConfig {
  task?: string;
  inputs?: Record<string, unknown>;
  browser_profile_id?: string | null;
  model?: string;
  max_steps?: number;
  timeout_seconds?: number;
  output_schema?: Record<string, unknown>;
  save_screenshots?: boolean;
}

/**
 * Image Gen Block Configuration
 */
export interface ImageGenBlockConfig extends BaseBlockConfig {
  model?: string;
  prompt?: string;
  size?: string;
  num_images?: number;
}

/**
 * Agent Block Configuration
 */
export interface AgentBlockConfig extends BaseBlockConfig {
  prompt?: string;
  model?: string;
  memory_bank_id?: string | null;
  tools?: string[];  // Array of tool names
  tool_connections?: Record<string, number | null>;  // Map tool name to connection_id (OAuth)
  tool_resource_ids?: Record<string, number | null>;  // Map tool name to integration_resource_id (Postgres, Supabase)
  max_iterations?: number;
  enable_artifacts?: boolean;
  output_schema?: Record<string, unknown>;  // Structured output JSON schema
}

/**
 * meetingsEA Bot Block Configuration
 */
export interface EABotBlockConfig extends BaseBlockConfig {
  meeting_url?: string;
  bot_name?: string;
  recording_mode?: 'speaker_view' | 'gallery_view' | 'audio_only';
  timeout_seconds?: number;
}

/**
 * Union type for all block configurations
 */
export type BlockConfig =
  | ForLoopBlockConfig
  | IfElseBlockConfig
  | ToolBlockConfig
  | McpBlockConfig
  | CodeBlockConfig
  | InputBlockConfig
  | HitlBlockConfig
  | BrowserBlockConfig
  | ImageGenBlockConfig
  | AgentBlockConfig
  | EABotBlockConfig;

/**
 * Props interface for block section components
 */
export interface BlockSectionProps<T extends BaseBlockConfig> {
  config: T;
  setConfig: Dispatch<SetStateAction<T>>;
  templateAutocomplete: TemplateAutocompleteControls;
  validationErrors?: Record<string, string>;
}
