import { Edge, Node } from '@xyflow/react';

import type { InputDef } from '@/types/workflow-spec';
import type { TriggerDescriptor } from '@/types/triggers';
import type { TriggerSpec } from '@/types/workflow-spec';

export type BlockType =
  | 'tool'
  | 'code'
  | 'if_else'
  | 'for_loop'
  | 'mcp'
  | 'hitl'
  | 'browser'
  | 'image_gen'
  | 'agent'
  | 'ea_bot';

export type CanvasNodeType = BlockType | 'trigger';

export interface FunctionBlockSchema {
  type: BlockType;
  label: string;
  category: string;
  description: string;
  defaults: Record<string, unknown>;
  config_schema: Record<string, unknown>;
  tags?: string[] | null;
}
export interface TriggerNodeHandlers {
  update?: (triggerId: string, payload: Partial<TriggerSpec>) => Promise<void>;
  toggle?: (triggerId: string, enabled: boolean) => Promise<void>;
  delete?: (triggerId: string) => Promise<void>;
}

export interface GmailIntegrationContext {
  ready: boolean;
  connectionId?: number | null;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
  triggerKey?: string;  // For account selector API
  onSelectAccount?: (connectionId: number | null) => void;  // For multi-account selection
}

export interface DiscordIntegrationContext {
  ready: boolean;
  connectionId?: number | null;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
  triggerKey?: string;
  onSelectAccount?: (connectionId: number | null) => void;
}

export interface SlackIntegrationContext {
  ready: boolean;
  connectionId?: number | null;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
  triggerKey?: string;
  onSelectAccount?: (connectionId: number | null) => void;
}

export interface SupabaseIntegrationContext {
  ready: boolean;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
}

export interface CalendarIntegrationContext {
  ready: boolean;
  connectionId?: number | null;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
  triggerKey?: string;
  onSelectAccount?: (connectionId: number | null) => void;
}

export interface GoogleSheetsIntegrationContext {
  ready: boolean;
  connectionId?: number | null;
  onConnect?: () => Promise<void> | void;
  isConnecting?: boolean;
  triggerKey?: string;
  onSelectAccount?: (connectionId: number | null) => void;
}

export interface TriggerNodeMeta {
  subscription?: TriggerSpec;      // Current trigger (no draft)
  descriptor?: TriggerDescriptor;
  workflowInputs?: Record<string, InputDef>;
  handlers: TriggerNodeHandlers;
  integration?: {
    gmail?: GmailIntegrationContext;
    discord?: DiscordIntegrationContext;
    slack?: SlackIntegrationContext;
    supabase?: SupabaseIntegrationContext;
    calendar?: CalendarIntegrationContext;
    google_sheets?: GoogleSheetsIntegrationContext;
  };
}

import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import type { NodeErrorState } from '@/stores/canvasStore';

export interface WorkflowNodeData extends Record<string, unknown> {
  type: CanvasNodeType;
  label: string;
  config?: ToolBlockConfig;
  oauth_scope?: string;
  selected?: boolean;
  onSelect?: () => void;
  triggerMeta?: TriggerNodeMeta;
  error?: NodeErrorState | null;
}

// Minimal payload used when creating/dropping a new block onto the canvas
export type DroppedBlockData = Pick<WorkflowNodeData, 'type' | 'label' | 'config'>;

export interface WorkflowNodeUpdateOptions {
  /**
   * When true, the caller expects the graph to be persisted immediately
   * (instead of waiting for the debounced autosave cycle).
   */
  persist?: boolean;
}

export type WorkflowEdgeData = {
  branch?: 'true' | 'false' | 'loop' | 'exit';
};

export type WorkflowEdge = Edge<WorkflowEdgeData>;

export function getNextBranchForSource(
  sourceId: string | null | undefined,
  nodes: Node<WorkflowNodeData>[],
  edges: WorkflowEdge[],
): 'true' | 'false' | 'loop' | 'exit' | undefined {
  if (!sourceId) {
    return undefined;
  }

  const sourceNode = nodes.find((node) => node.id === sourceId);
  if (!sourceNode) {
    return undefined;
  }

  const outgoing = edges.filter((edge) => edge.source === sourceId);
  if (sourceNode.type === 'if_else') {
    const hasTrue = outgoing.some((edge) => edge.data?.branch === 'true');
    const hasFalse = outgoing.some((edge) => edge.data?.branch === 'false');
    if (!hasTrue) {
      return 'true';
    }
    if (!hasFalse) {
      return 'false';
    }
    return undefined;
  }

  if (sourceNode.type === 'for_loop') {
    const hasLoop = outgoing.some((edge) => edge.data?.branch === 'loop');
    const hasExit = outgoing.some((edge) => edge.data?.branch === 'exit');
    if (!hasLoop) {
      return 'loop';
    }
    if (!hasExit) {
      return 'exit';
    }
    return undefined;
  }

  return undefined;
}
