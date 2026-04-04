/**
 * TypeScript mirror of the backend WorkflowSpec schema.
 *
 * We only model the fields we actually need on the frontend so that
 * conversions between ReactFlow graph data and backend contracts stay typed.
 */

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;
export interface JsonObject {
  [key: string]: JsonValue;
}
export type JsonArray = JsonValue[];

export interface SchemaRef {
  id: string;
}

export interface InlineSchema {
  json_schema: JsonObject;
}

export interface InlineSchemaAlias {
  schema: JsonObject;
}

export type SchemaSpec = SchemaRef | InlineSchema | InlineSchemaAlias | JsonObject;

export interface OutputContract {
  mode: 'text' | 'json';
  schema?: SchemaSpec;
}

export interface InputDef {
  type: 'string' | 'integer' | 'number' | 'boolean' | 'object' | 'array';
  description?: string;
  default?: JsonValue;
  required?: boolean;
}
// Inputs are no longer persisted on the workflow spec. We keep the definition
// type around for components that still render manual run fields.

export interface TriggerMetadata {
  sample_event?: JsonObject;
  requires_connection?: boolean;
}

export interface TriggerUIMeta extends JsonObject {
  // Subscription enrichment fields (server-provided)
  secret_token?: string | null;
  webhook_url?: string | null;
  form_url?: string | null;
  created_at?: string;
  updated_at?: string;
  // Allow other UI metadata
  [key: string]: JsonValue;
}

export interface TriggerSpec {
  // Unique instance identifier (allows multiple triggers of same type)
  id: string;

  // Trigger type and mode
  key: string;
  mode: string;

  // Event schema (defines the structure of trigger events)
  event_schema: JsonObject;

  // Metadata
  meta: TriggerMetadata;

  // Configuration
  filters?: JsonObject;
  provider_config?: JsonObject | null;
  ui_meta?: TriggerUIMeta;
}

export interface NodeMeta extends JsonObject {
  reactflow?: JsonObject;
}

interface NodeBase {
  id: string;
  ui?: NodeMeta;
}

export interface TaskNode extends NodeBase {
  type: 'task';
  kind: 'set';
  value: JsonValue;
  inputs?: JsonObject;
  outputs?: OutputContract | null;
}

export interface ToolNode extends NodeBase {
  type: 'tool';
  tool: string;
  inputs?: JsonObject;
  expect_outputs?: OutputContract | null;
  connection_id?: number | null;  // Database ID of OAuth connection for multi-account support
}


export interface IfNode extends NodeBase {
  type: 'if';
  condition: string;
  // Note: then/else branches are now defined by edges with type conditional_true/conditional_false
}

export interface ForEachNode extends NodeBase {
  type: 'for_each';
  items: string;
  item_var?: string;
  index_var?: string;
  outputs?: OutputContract | null;
  // Note: loop body is now defined by edges with type loop_body, exit by loop_exit
}

export interface McpNode extends NodeBase {
  type: 'mcp';
  server: string;
  server_type: 'http' | 'stdio';
  tool: string;
  inputs?: JsonObject;
  auth?: JsonObject | null;
  mcp_server_config_id?: number | null;
  expect_outputs?: OutputContract | null;
}

export interface HitlDeliveryChannel {
  type: 'platform' | 'gmail';
  gmail?: { recipient_email: string };
}

export interface HitlNode extends NodeBase {
  type: 'hitl';
  title: string;
  description?: string;
  display?: Array<{ label: string; value: string }>;
  inputs: Array<{
    id: string;
    question: string;
    input_type: 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean';
    options?: Array<{ value: string; label: string; requires_text?: boolean }>;
    required?: boolean;
    placeholder?: string;
    default_value?: string | number | boolean;
  }>;
  timeout_seconds?: number;
  /** Channels for delivering HITL notifications. Default: platform only */
  delivery_channels?: HitlDeliveryChannel[];
}

export interface BrowserNode extends NodeBase {
  type: 'browser';
  task: string;
  inputs?: JsonObject;
  browser_profile_id?: string | null;
  model?: string;
  max_steps?: number;
  timeout_seconds?: number;
  expect_outputs?: OutputContract | null;
  save_screenshots?: boolean;
}

export interface ImageGenNode extends NodeBase {
  type: 'image_gen';
  inputs: JsonObject;
}

export interface AgentNode extends NodeBase {
  type: 'agent';
  inputs: JsonObject;  // Contains model, prompt, memory_bank_id, tools, max_iterations
  outputs?: OutputContract;
}

export interface EABotNode extends NodeBase {
  type: 'ea_bot';
  inputs: JsonObject;  // Contains meeting_url, bot_name, recording_mode
  timeout_seconds?: number;
}

export type WorkflowNode = TaskNode | ToolNode | IfNode | ForEachNode | McpNode | HitlNode | BrowserNode | ImageGenNode | AgentNode | EABotNode;

/**
 * Edge types for the workflow graph (backend spec format).
 * Maps to the backend EdgeType enum.
 */
export type EdgeType =
  | 'default'
  | 'conditional_true'
  | 'conditional_false'
  | 'loop_body'
  | 'loop_exit'
  | 'trigger';

/**
 * Workflow spec edge (backend format).
 * Different from WorkflowEdge which is the ReactFlow frontend format.
 */
export interface WorkflowSpecEdge {
  source: string;
  target: string;
  type: EdgeType;
  ui?: JsonObject;
}

export interface WorkflowSpec {
  version: string;
  nodes: WorkflowNode[];
  edges: WorkflowSpecEdge[];
  triggers: TriggerSpec[];
}

export interface WorkflowVersionSummary {
  version_id: number;
  status: string;
  version_number?: number;
  created_at: string;
}

export interface WorkflowVersionListItem extends WorkflowVersionSummary {
  is_latest: boolean;
  is_published: boolean;
}

export interface WorkflowVersionListResponse {
  workflow_id: string;
  versions: WorkflowVersionListItem[];
  latest_version_id?: number | null;
  published_version_id?: number | null;
}

export type WorkflowVersionRestoreRequest = Record<string, never>;

export interface WorkflowSummary {
  workflow_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  integrations?: string[];
}

export interface WorkflowResponse extends WorkflowSummary {
  spec: WorkflowSpec;
}

export interface WorkflowListResponse {
  items: WorkflowSummary[];
  next_cursor?: string | null;
}

export interface WorkflowCreateRequest {
  name: string;
  spec: WorkflowSpec;
}

export interface WorkflowUpdateRequest {
  name?: string;
}

export interface WorkflowDraftPatchRequest {
  spec: WorkflowSpec;
}

export interface WorkflowPublishRequest {
  version_id: number;
}

export interface RunFromWorkflowRequest {
  version?: number;
  inputs?: Record<string, JsonValue>;
  config?: Record<string, JsonValue>;
}

export interface RunFromSpecRequest {
  spec: WorkflowSpec;
  inputs?: Record<string, JsonValue>;
  config?: Record<string, JsonValue>;
}

export interface RunResponse {
  run_id: string;
  status: string;
  workflow_version_id?: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress?: {
    completed: number;
    total: number;
  } | null;
  current_node_id?: string | null;
  last_error?: string | null;
}

export interface RunResultResponse {
  run_id: string;
  status: string;
  output?: JsonObject | null;
  state?: JsonObject | null;
  metrics?: JsonObject | null;
}
