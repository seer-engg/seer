import { useMemo } from 'react';
import { Node } from '@xyflow/react';
import { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import { normalizeEdges, normalizeNodes } from '@/lib/workflow-normalization';
import type { PreviewGraphData } from './WorkflowPanels';

interface ProposalPreviewData {
  graph?: {
    nodes?: unknown[];
    edges?: unknown[];
  };
}

/**
 * Derives the provider name from a trigger key.
 * e.g., 'poll.gmail.email_received' -> 'google'
 *       'poll.discord.message_received' -> 'discord'
 */
export function getProviderFromTriggerKey(triggerKey: string): string {
  if (triggerKey.includes('gmail')) return 'google';
  if (triggerKey.includes('google_calendar')) return 'google';
  if (triggerKey.includes('google_sheets')) return 'google';
  if (triggerKey.includes('discord')) return 'discord';
  if (triggerKey.startsWith('form.')) return 'form';
  // Extract provider from key pattern: mode.provider.event
  const parts = triggerKey.split('.');
  return parts[1] ?? 'unknown';
}

/**
 * Parses a connection ID string (e.g., "gmail:123") to extract the numeric ID.
 * Returns null if the input is invalid.
 */
export function parseConnectionId(raw: string | null): number | null {
  if (!raw) return null;
  const segments = raw.split(':');
  const numeric = Number(segments[segments.length - 1]);
  return Number.isNaN(numeric) ? null : numeric;
}

export function usePreviewGraph(proposalPreview: unknown, functionBlocksMap: Record<string, unknown>): PreviewGraphData | null {
  return useMemo(() => {
    if (!proposalPreview) return null;
    const preview = proposalPreview as ProposalPreviewData;
    const previewNodes = normalizeNodes(preview.graph?.nodes ?? [], functionBlocksMap);
    const previewEdges = normalizeEdges(preview.graph?.edges ?? []);
    return { nodes: previewNodes, edges: previewEdges };
  }, [proposalPreview, functionBlocksMap]);
}

interface TriggerHandlersConfigParams {
  updateTrigger: (triggerId: string, payload: Record<string, unknown>) => Promise<unknown>;
  toggleTrigger: (triggerId: string, enabled: boolean) => Promise<unknown>;
  deleteTrigger: (triggerId: string) => Promise<unknown>;
}

export function useTriggerHandlersConfig({
  updateTrigger,
  toggleTrigger,
  deleteTrigger,
}: TriggerHandlersConfigParams) {
  return useMemo(
    () => ({
      update: (triggerId: string, payload: Record<string, unknown>) =>
        updateTrigger(triggerId, payload).then(() => undefined),
      toggle: (triggerId: string, enabled: boolean) =>
        toggleTrigger(triggerId, enabled).then(() => undefined),
      delete: (triggerId: string) => deleteTrigger(triggerId).then(() => undefined),
    }),
    [updateTrigger, toggleTrigger, deleteTrigger],
  );
}

export function buildCanvasProps(params: {
  selectedWorkflowId: string; previewGraph: PreviewGraphData | null; proposalPreview: unknown;
  lifecycleStatus: unknown; workflowVersions: unknown[]; isLoadingWorkflowVersions: boolean;
  isExecuting: boolean; isPublishing: boolean; isRestoringVersion: boolean; canRun: boolean;
  canPublish: boolean; publishDisabledReason: string; runDisabledReason: string; isImportDialogOpen: boolean;
  handleCanvasNodeClick: (id: string) => void; handleNodeDrop: (e: unknown) => void;
  handleRunClick: () => void; handlePublish: () => void;
  handleRestoreVersion: (versionId: number) => void;
  setImportDialogOpen: (open: boolean) => void;
  handleImportWorkflow: (file: File) => void;
  readOnly?: boolean;
  readOnlyReason?: string | null;
  onRetryLock?: () => void;
  versionRestoreDisabledReason?: string;
}) {
  const {
    selectedWorkflowId, previewGraph, proposalPreview, lifecycleStatus,
    workflowVersions, isLoadingWorkflowVersions, isExecuting, isPublishing, isRestoringVersion,
    canRun, canPublish, publishDisabledReason, runDisabledReason, isImportDialogOpen,
    handleCanvasNodeClick, handleNodeDrop, handleRunClick, handlePublish, handleRestoreVersion,
    setImportDialogOpen, handleImportWorkflow, readOnly, readOnlyReason, onRetryLock,
    versionRestoreDisabledReason,
  } = params;
  return {
    selectedWorkflowId, previewGraph, proposalPreview, lifecycleStatus,
    workflowVersions, isLoadingWorkflowVersions, isExecuting, isPublishing, isRestoringVersion,
    canRun, canPublish, publishDisabledReason, runDisabledReason, isImportDialogOpen,
    onNodeClick: handleCanvasNodeClick, onNodeDrop: handleNodeDrop,
    onRunClick: handleRunClick,
    onPublishClick: handlePublish,
    onVersionRestore: handleRestoreVersion, onImportWorkflow: handleImportWorkflow,
    setImportDialogOpen,
    readOnly,
    readOnlyReason,
    onRetryLock,
    versionRestoreDisabledReason,
  };
}

export function buildProps(params: {
  handleBlockSelect: (id: string) => void; selectedWorkflowId: string;
  availableBlocks: unknown[]; triggerOptions: unknown;
  isLoadingTriggers: boolean; selectedNode: Node<WorkflowNodeData> | null; allNodes: Node<WorkflowNodeData>[];
  allEdges: WorkflowEdge[]; handleNodeConfigUpdate: (nodeId: string, updates: Partial<WorkflowNodeData>, options?: unknown) => Promise<void> | void;
  handleClearSelection: () => void; workflowInputsDef: Record<string, unknown>;
  workflowTriggers: unknown[]; isPreviewActive: boolean;
}) {
  return {
    onBlockSelect: params.handleBlockSelect, workflowId: params.selectedWorkflowId,
    functionBlocks: params.availableBlocks,
    triggerOptions: params.triggerOptions, isLoadingTriggers: params.isLoadingTriggers, triggerInfoMessage: undefined,
    selectedNode: params.selectedNode, allNodes: params.allNodes, allEdges: params.allEdges,
    onNodeConfigUpdate: params.handleNodeConfigUpdate, onClearSelection: params.handleClearSelection,
    workflowInputs: params.workflowInputsDef, triggers: params.workflowTriggers, readOnly: params.isPreviewActive,
  };
}
