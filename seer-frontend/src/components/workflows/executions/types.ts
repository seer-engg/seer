import type { JsonObject } from '@/types/workflow-spec';
export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted';

export interface WorkflowRunSummary {
  run_id: string;
  status: RunStatus;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  inputs?: JsonObject | null;
  output?: JsonObject | null;
  error?: string | null;
}

export interface WorkflowRunListResponse {
  workflow_id: string;
  runs: WorkflowRunSummary[];
}

export interface RunStatusResponse {
  run_id: string;
  status: RunStatus;
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

export interface WorkflowFileRef {
  _type: 'workflow_file_ref';
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  workflow_run_id: string;
  created_at: string;
  md5_hash?: string;
}

export interface AgentToolCall {
  tool: string;
  input: Record<string, unknown>;
}

export interface AgentStep {
  type: 'reasoning' | 'tool_response';
  content: string;
  tool?: string;
  tool_calls?: AgentToolCall[];
}

export interface WorkflowNodeTrace {
  node_id: string;
  node_type: string;
  inputs?: JsonObject | null;
  output?: unknown;
  timestamp?: string;
  output_key?: string;
  tool_name?: string;
  model?: string;
  error?: string | { type: string; message: string };
  is_synthetic?: boolean; // Mark generated nodes (e.g., trigger/init)
  artifacts?: WorkflowFileRef[];
  steps?: AgentStep[];
  iterations?: number;
  prompt?: string;
}

export interface TriggerInfo {
  source: string;
  trigger_id: string;
  trigger_key: string;
  title: string;
  event_data?: Record<string, unknown> | null;
  occurred_at?: string | null;
  received_at?: string | null;
}

export interface RunHistoryEntry {
  run_id: string;
  workflow_id: string | null;
  status: RunStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  inputs?: JsonObject | null;
  output?: JsonObject | null;
  error?: string | null;
  trigger?: TriggerInfo | null;
  nodes: WorkflowNodeTrace[];
  execution_graph?: {
    nodes: Array<{ id: string; type: string; label: string; position?: { x: number; y: number } }>;
    edges: Array<{ source: string; target: string; label?: string }>;
  };
}

export interface RunHistoryResponse {
  run_id: string;
  history: RunHistoryEntry[];
}

/**
 * HITL interrupt data returned when a workflow is in 'interrupted' status.
 * This matches the flat structure returned by GET /runs/{run_id}/interrupt
 */
export interface HitlInterruptData {
  run_id: string;
  status: string;
  type?: 'hitl' | 'browser_hitl';
  node_id: string;
  title: string;
  description?: string;
  session_id?: string; // Browser pool session ID for live view (browser_hitl only)
  display: Array<{ label: string; value: unknown }>;
  inputs: Array<{
    id: string;
    question: string;
    input_type: 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean' | 'table';
    options?: Array<{ value: string; label: string; requires_text?: boolean }>;
    required?: boolean;
    placeholder?: string | null;
    default_value?: unknown;
    columns?: Array<{
      id: string;
      header: string;
      input_type: 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean';
      options?: Array<{ value: string; label: string }>;
      required?: boolean;
    }>;
    rows?: Array<{
      display: Record<string, unknown>;
      row_index: number;
      row_data: unknown;
    }>;
  }>;
  timeout_seconds?: number;
  expires_at?: string;
  is_expired?: boolean;
}

/**
 * Payload for resuming an interrupted workflow with user responses
 */
export interface HitlResumePayload {
  responses: Record<string, string | number | boolean | string[] | Array<Record<string, unknown>>>;
}

export interface ExecutionNavigationState {
  workflowId?: string | null;
  workflowName?: string | null;
}
