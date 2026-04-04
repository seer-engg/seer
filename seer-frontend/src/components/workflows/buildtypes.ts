import type { ReactNode } from 'react';
import { Node } from '@xyflow/react';

import { WorkflowEdge, WorkflowNodeData } from './types';
import type { JsonObject, WorkflowSpec, WorkflowVersionSummary } from '@/types/workflow-spec';
import type { ClarificationQuestion, ClarificationQuestions } from '@/types/discovery';

export interface Tool {
  name: string;
  description: string;
  toolkit?: string;
  slug?: string;
  provider?: string;
  integration_type?: string;
  output_schema?: JsonObject | null;
}

export interface BuiltInBlock {
  type: string;
  label: string;
  description: string;
  icon: ReactNode;
}

export interface UserSummary {
  user_id: string;
  email?: string | null;
  full_name?: string | null;
}

export interface WorkflowGraphPayload {
  nodes?: Node<WorkflowNodeData & Record<string, unknown>>[];
  edges?: WorkflowEdge[];
}

export interface WorkflowProposal {
  id: number;
  workflow_id: string;
  session_id?: number | null;
  created_by: UserSummary;
  summary: string;
  status: 'pending' | 'accepted' | 'rejected';
  spec: WorkflowSpec;
  preview_graph?: JsonObject | null;
  applied_graph?: JsonObject | null;
  metadata?: JsonObject | null;
  decided_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface WorkflowProposalActionResponse {
  proposal: WorkflowProposal;
  workflow_graph?: WorkflowGraphPayload | null;
}

export interface WorkflowProposalPreview {
  proposal: WorkflowProposal;
  graph: WorkflowGraphPayload;
}

export interface WorkflowLifecycleStatus {
  lastTestedVersionId?: number | null;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  proposal?: WorkflowProposal | null;
  proposalError?: string | null;
  thinking?: string[];
  timestamp: Date;
  model?: string;
  interruptRequired?: boolean;
  interruptData?: {
    type?: string;
    clarification_question?: ClarificationQuestion;
    clarification_questions?: ClarificationQuestions;
    [key: string]: unknown;
  };
  error?: ChatErrorMetadata;
}

export interface ChatErrorMetadata {
  type: 'cost_cap_exceeded' | 'timeout' | 'generic';
  details?: {
    accumulated_cost?: number;
    cost_cap?: number;
    run_type?: 'chat' | 'workflow';
  };
}

/**
 * Chat execution status for async mode
 */
export type ChatExecutionStatus = 'queued' | 'running' | 'completed' | 'failed' | 'interrupted';

/**
 * Response from polling the /chat/status/{session_id} endpoint
 * Used to check execution status and retrieve results in async mode
 */
export interface ChatStatusResponse {
  status: ChatExecutionStatus;
  session_id: number;
  thread_id: string;

  // Available when completed
  response?: string;
  thinking?: string[];
  proposal?: WorkflowProposal | null;

  // Available when interrupted (clarification question)
  interrupt_required?: boolean;
  interrupt_data?: {
    type?: string;
    clarification_question?: ClarificationQuestion;
    clarification_questions?: ClarificationQuestions;
    [key: string]: unknown;
  };

  // Available when failed
  error?: {
    detail?: string;
    reason?: string;
    [key: string]: unknown;
  };

  // Timing information
  started_at?: string;
  finished_at?: string;
}

export interface ChatSession {
  id: number;
  workflow_id: string;
  user_id?: string;
  thread_id: string;
  title?: string;
  created_at: string;
  updated_at: string;

  // Async execution tracking (added for resilient chat APIs)
  current_execution_status?: ChatExecutionStatus;
  current_execution_task_id?: string;
  pending_interrupt_type?: string | null;
  pending_interrupt_data?: {
    type?: string;
    clarification_question?: ClarificationQuestion;
    clarification_questions?: ClarificationQuestions;
    questions?: ClarificationQuestion[];
    [key: string]: unknown;
  } | null;
}

export interface ModelInfo {
  id: string;
  provider: 'openai' | 'anthropic' | 'openrouter' | 'google' | 'custom';
  name: string;
  available: boolean;
}

export interface ModelDescriptor {
  id: string;
  title: string;
  supports_json_schema: boolean;
  /** Whether the model supports vision/image inputs */
  supports_vision?: boolean;
}

// Unified item types for the Build Panel
export type ItemType = 'block' | 'trigger' | 'action';

export interface BaseItem {
  id: string;
  type: ItemType;
  label: string;
  description?: string;
  icon?: ReactNode;
  searchTerms: string;
}

export interface BlockItem extends BaseItem {
  type: 'block';
  blockType: string;
  builtInBlock: BuiltInBlock;
}

export interface TriggerItem extends BaseItem {
  type: 'trigger';
  triggerKey: string;
  status?: 'ready' | 'action-required';
  badge?: string;
  actionLabel?: string;
  secondaryActionLabel?: string;
  disabled?: boolean;
  disabledReason?: string;
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
  isPrimaryActionLoading?: boolean;
  isSecondaryActionLoading?: boolean;
}

export interface ActionItem extends BaseItem {
  type: 'action';
  tool?: Tool; // Legacy support for individual tools
  integrationType: string;
  integrationGroup?: {
    integration_type: string;
    display_name: string;
    tools: Tool[];
    isConnected: boolean;
  };
  toolCount?: number; // Number of tools in the integration group
}

export type UnifiedItem = BlockItem | TriggerItem | ActionItem;

// ============================================================================
// SSE Streaming Event Types
// ============================================================================

export type StreamEventType =
  | 'session_info'
  | 'agent_start'
  | 'tool_start'
  | 'tool_end'
  | 'ai_message'
  | 'agent_end'
  | 'interrupt'
  | 'error'
  | 'done';

export interface StreamEvent {
  type: StreamEventType;
  data: Record<string, unknown>;
  session_id: number;
}

/** A single event in the agent's activity chain during streaming */
export type StreamActivity =
  | { type: 'tool'; tool: string; status: 'running' | 'done' }
  | { type: 'ai_message'; content: string };
