import { useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { useCanvasStore, useUIStore, useWorkflowStore } from '@/stores';
import { useFunctionBlocksQuery } from './useFunctionBlocksQuery';
import { useWorkflowDetailQuery, useWorkflowListQuery } from './useWorkflowQueries';

export function useWorkflowPageState() {
  const { workflowId: urlWorkflowId } = useParams<{ workflowId?: string }>();
  const workflowsQuery = useWorkflowListQuery();
  const workflowDetailQuery = useWorkflowDetailQuery(urlWorkflowId);
  const functionBlocksQuery = useFunctionBlocksQuery();

  // Canvas state
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const editingNodeId = useCanvasStore((state) => state.editingNodeId);
  const isDirty = useCanvasStore((state) => state.isDirty);
  const setNodes = useCanvasStore((state) => state.setNodes);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const setSelectedNodeId = useCanvasStore((state) => state.setSelectedNodeId);
  const setEditingNodeId = useCanvasStore((state) => state.setEditingNodeId);
  const autosaveStatus = useCanvasStore((state) => state.autosaveStatus);

  // UI state
  const isConfigDialogOpen = useUIStore((state) => state.isConfigDialogOpen);
  const isImportDialogOpen = useUIStore((state) => state.isImportDialogOpen);
  const isKeymapOpen = useUIStore((state) => state.isKeymapOpen);
  const isInputDialogOpen = useUIStore((state) => state.isInputDialogOpen);
  const buildChatPanelCollapsed = useUIStore((state) => state.buildChatPanelCollapsed);
  const proposalPreview = useUIStore((state) => state.proposalPreview);
  const setConfigDialogOpen = useUIStore((state) => state.setConfigDialogOpen);
  const setImportDialogOpen = useUIStore((state) => state.setImportDialogOpen);
  const setKeymapOpen = useUIStore((state) => state.setKeymapOpen);
  const setInputDialogOpen = useUIStore((state) => state.setInputDialogOpen);
  const setProposalPreview = useUIStore((state) => state.setProposalPreview);
  const isRunOptionsDialogOpen = useUIStore((state) => state.isRunOptionsDialogOpen);
  const isTriggerEventPickerOpen = useUIStore((state) => state.isTriggerEventPickerOpen);
  const selectedTriggerForPicker = useUIStore((state) => state.selectedTriggerForPicker);
  const setRunOptionsDialogOpen = useUIStore((state) => state.setRunOptionsDialogOpen);
  const setTriggerEventPickerOpen = useUIStore((state) => state.setTriggerEventPickerOpen);
  const setSelectedTriggerForPicker = useUIStore((state) => state.setSelectedTriggerForPicker);

  // Workflow state
  const setWorkflowName = useWorkflowStore((state) => state.setWorkflowName);
  const selectedWorkflowId = useWorkflowStore((state) => state.selectedWorkflowId);
  const setSelectedWorkflowId = useWorkflowStore((state) => state.setSelectedWorkflowId);
  const inputData = useWorkflowStore((state) => state.workflowInputData);
  const setInputData = useWorkflowStore((state) => state.setWorkflowInputData);
  const isLoadingWorkflow = useWorkflowStore((state) => state.isLoadingWorkflow);
  const setIsLoadingWorkflow = useWorkflowStore((state) => state.setIsLoadingWorkflow);
  const isExecuting = useWorkflowStore((state) => state.isExecuting);
  const isPublishing = useWorkflowStore((state) => state.isPublishing);
  const isRestoringVersion = useWorkflowStore((state) => state.isRestoringVersion);

  const editingNode = useMemo(
    () => nodes.find((node) => node.id === editingNodeId) ?? null,
    [nodes, editingNodeId],
  );

  return {
    urlWorkflowId,
    nodes,
    edges,
    editingNodeId,
    isDirty,
    setNodes,
    setEdges,
    setSelectedNodeId,
    setEditingNodeId,
    autosaveStatus,
    isConfigDialogOpen,
    isImportDialogOpen,
    isKeymapOpen,
    isInputDialogOpen,
    buildChatPanelCollapsed,
    proposalPreview,
    setConfigDialogOpen,
    setImportDialogOpen,
    setKeymapOpen,
    setInputDialogOpen,
    setProposalPreview,
    isRunOptionsDialogOpen,
    isTriggerEventPickerOpen,
    selectedTriggerForPicker,
    setRunOptionsDialogOpen,
    setTriggerEventPickerOpen,
    setSelectedTriggerForPicker,
    setWorkflowName,
    selectedWorkflowId,
    setSelectedWorkflowId,
    inputData,
    setInputData,
    isLoadingWorkflow,
    setIsLoadingWorkflow,
    loadedWorkflow: workflowDetailQuery.data ?? null,
    workflowError: workflowDetailQuery.error ?? null,
    workflowListError: workflowsQuery.error ?? null,
    workflows: workflowsQuery.data ?? [],
    workflowsLoaded: workflowsQuery.status === 'success',
    isLoadingWorkflows: workflowsQuery.isLoading,
    isFetchingWorkflow: workflowDetailQuery.isFetching,
    isExecuting,
    isPublishing,
    isRestoringVersion,
    functionBlocksReady: functionBlocksQuery.status !== 'pending',
    functionBlockSchemas: functionBlocksQuery.functionBlocks,
    functionBlocksMap: functionBlocksQuery.functionBlocksMap,
    editingNode,
  };
}
