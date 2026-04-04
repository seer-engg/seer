import { useCallback, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueryClient } from '@tanstack/react-query';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import type { WorkflowListItem } from '@/stores/workflowStore';
import { toast } from '@/components/ui/sonner';
import { BackendAPIError, extractProblemDetails, extractNodeErrors } from '@/lib/api-client';
import { showError } from '@/lib/error-handler';
import { workflowKeys, workflowRunKeys } from '@/lib/query-keys';
import { getWorkflowDetail, toWorkflowModel } from '@/lib/workflows-api';
import { useFixWithAI } from './useFixWithAI';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useCanvasStore, useUIStore } from '@/stores';
import { useTriggersStore } from '@/stores/triggersStore';
import { handleDraftConflict } from '../components/workflows/canvas/conflictHandler';
import { normalizeNodes, normalizeEdges } from '@/lib/workflow-normalization';
import { generateWorkflowName } from '@/lib/workflow-names';
import { useFunctionBlocksQuery } from './useFunctionBlocksQuery';
import { useWorkflowDetailQuery } from './useWorkflowQueries';
import { useActiveWorkflowId } from './useActiveWorkflowId';

function buildErrorContext(error: unknown): string {
  const problemDetails = extractProblemDetails(error);
  const nodeErrors = extractNodeErrors(error);

  if (!problemDetails) return String(error);

  let context = `${problemDetails.title ?? 'Error'}: ${problemDetails.detail ?? ''}`.trim();
  if (nodeErrors.length > 0) {
    context += '\n\nNode errors:\n' + nodeErrors.map((e) => `- ${e.node_id}: ${e.message}`).join('\n');
  }
  return context;
}

// eslint-disable-next-line max-lines-per-function -- Workflow page lifecycle actions remain consolidated during refactor
export function useWorkflowActions() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { fixWithAI } = useFixWithAI();

  // Fetch required state from stores - FIXED: Individual selectors instead of useShallow
  const activeWorkflowId = useActiveWorkflowId();
  const workflowName = useWorkflowStore((state) => state.workflowName);
  const setWorkflowName = useWorkflowStore((state) => state.setWorkflowName);
  const setSelectedWorkflowId = useWorkflowStore((state) => state.setSelectedWorkflowId);
  const createWorkflow = useWorkflowStore((state) => state.createWorkflow);
  const saveWorkflowDraft = useWorkflowStore((state) => state.saveWorkflowDraft);
  const executeWorkflow = useWorkflowStore((state) => state.executeWorkflow);
  const publishWorkflow = useWorkflowStore((state) => state.publishWorkflow);
  const restoreWorkflowVersion = useWorkflowStore((state) => state.restoreWorkflowVersion);
  const deleteWorkflow = useWorkflowStore((state) => state.deleteWorkflow);
  const updateWorkflowMetadata = useWorkflowStore((state) => state.updateWorkflowMetadata);
  const exportWorkflow = useWorkflowStore((state) => state.exportWorkflow);
  const importWorkflow = useWorkflowStore((state) => state.importWorkflow);
  const { data: loadedWorkflow = null } = useWorkflowDetailQuery(activeWorkflowId);

  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const setNodes = useCanvasStore((state) => state.setNodes);
  const setEdges = useCanvasStore((state) => state.setEdges);
  const setAutosaveStatus = useCanvasStore((state) => state.setAutosaveStatus);
  const setNodeErrors = useCanvasStore((state) => state.setNodeErrors);
  const clearNodeErrors = useCanvasStore((state) => state.clearNodeErrors);

  const { functionBlocksMap } = useFunctionBlocksQuery();

  const lastRunVersionId = useUIStore((state) => state.lastRunVersionId);
  const setLastRunVersionId = useUIStore((state) => state.setLastRunVersionId);
  const resetSavedDataRef = useRef<(() => void) | undefined>();

  const reloadWorkflow = useCallback(
    async (workflowId: string) => {
      const response = await queryClient.fetchQuery({
        queryKey: workflowKeys.detail(workflowId),
        staleTime: 0,
        queryFn: async () => toWorkflowModel(await getWorkflowDetail(workflowId)),
      });
      useTriggersStore.getState().loadTriggersFromSpec(workflowId, response.spec.triggers ?? []);
      return response;
    },
    [queryClient],
  );

  // ==================== LIFECYCLE ACTIONS ====================

  /**
   * Handle draft conflicts when concurrent edits occur
   */
  const handleDraftConflictCallback = useCallback(async () => {
    if (!activeWorkflowId) return;

    await handleDraftConflict({
      selectedWorkflowId: activeWorkflowId,
      reloadWorkflow,
      setNodes,
      setEdges,
      functionBlocksMap,
      setLastRunVersionId,
      resetSavedDataRef,
      invalidateWorkflowVersions: () => {
        void queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(activeWorkflowId) });
      },
    });
  }, [activeWorkflowId, reloadWorkflow, setNodes, setEdges, functionBlocksMap, setLastRunVersionId, queryClient]);

  /**
   * Save or create workflow
   */
  const handleSave = useCallback(async () => {
    // Clear node errors on save
    clearNodeErrors();

    const graph = { nodes, edges };

    // Create new workflow if none selected
    if (!activeWorkflowId) {
      try {
        const newWorkflow = await createWorkflow(workflowName || generateWorkflowName(), graph);
        setSelectedWorkflowId(newWorkflow.workflow_id);
        setWorkflowName(newWorkflow.name);
        setLastRunVersionId(null);
        navigate(`/workflows/${newWorkflow.workflow_id}`, { replace: true });
        toast.success('Workflow created');
      } catch (error) {
        console.error('Failed to create workflow:', error);
        toast.error('Failed to create workflow');
        throw error;
      }
      return;
    }

    // Save existing workflow
    if (!loadedWorkflow) {
      toast.error('Workflow is still loading. Please try again in a moment.');
      return;
    }

    try {
      await saveWorkflowDraft(activeWorkflowId, {
        graph,
      });
      setLastRunVersionId(null);
      toast.success('Workflow saved');
    } catch (error) {
      if (error instanceof BackendAPIError && error.status === 409) {
        await handleDraftConflictCallback();
        return;
      }
      console.error('Failed to save workflow:', error);
      toast.error('Failed to save workflow');
      throw error;
    }
  }, [
    nodes,
    edges,
    activeWorkflowId,
    workflowName,
    loadedWorkflow,
    createWorkflow,
    saveWorkflowDraft,
    setSelectedWorkflowId,
    setWorkflowName,
    navigate,
    handleDraftConflictCallback,
    setLastRunVersionId,
    clearNodeErrors,
  ]);

  /**
   * Execute workflow test
   */
  const runWorkflowTest = useCallback(
    async (inputs?: Record<string, unknown>) => {
      if (!activeWorkflowId) {
        toast.error('Please save the workflow first');
        return false;
      }

      // Clear previous errors before new execution
      clearNodeErrors();

      try {
        const result = await executeWorkflow(activeWorkflowId, inputs ?? {});
        if (result.workflow_version_id) {
          setLastRunVersionId(result.workflow_version_id);
          toast.success('Workflow test started', {
            description: `Run ID: ${result.run_id.slice(0, 8)}...`,
          });
        }

        // Invalidate executions list to show new run immediately
        await queryClient.invalidateQueries({
          queryKey: workflowRunKeys.list(activeWorkflowId),
        });

        return true;
      } catch (error) {
        console.error('Failed to execute workflow:', error);

        // Extract and set node errors for highlighting
        const nodeErrors = extractNodeErrors(error);
        if (nodeErrors.length > 0) {
          setNodeErrors(
            nodeErrors.map((e) => ({
              nodeId: e.node_id,
              code: e.code,
              message: e.message,
              location: e.location,
              expression: e.expression,
            }))
          );
        }

        // Extract detailed error info for workflow validation errors
        const problemDetails = extractProblemDetails(error);

        const errorContext = buildErrorContext(error);
        if (problemDetails) {
          toast.error(problemDetails.title || 'Workflow validation failed', {
            description: problemDetails.detail,
            duration: Infinity, // Stay until user dismisses
            action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
          });
        } else {
          toast.error('Failed to start workflow test', {
            duration: Infinity,
            action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
          });
        }

        return false;
      }
    },
    [activeWorkflowId, executeWorkflow, setLastRunVersionId, queryClient, setNodeErrors, clearNodeErrors, fixWithAI],
  );

  /**
   * Execute workflow with a real trigger event (for testing with actual data)
   */
  const runWorkflowWithTriggerEvent = useCallback(
    async (triggerId: string, envelope: Record<string, unknown>) => {
      if (!activeWorkflowId) {
        toast.error('Please save the workflow first');
        return false;
      }

      // Clear previous errors before new execution
      clearNodeErrors();

      try {
        const result = await executeWorkflow(activeWorkflowId, {}, {}, { triggerId, envelope });
        if (result.workflow_version_id) {
          setLastRunVersionId(result.workflow_version_id);
          toast.success('Workflow test started with real event', {
            description: `Run ID: ${result.run_id.slice(0, 8)}...`,
          });
        }

        // Invalidate executions list to show new run immediately
        await queryClient.invalidateQueries({
          queryKey: workflowRunKeys.list(activeWorkflowId),
        });

        return true;
      } catch (error) {
        console.error('Failed to execute workflow with trigger event:', error);

        // Extract and set node errors for highlighting
        const nodeErrors = extractNodeErrors(error);
        if (nodeErrors.length > 0) {
          setNodeErrors(
            nodeErrors.map((e) => ({
              nodeId: e.node_id,
              code: e.code,
              message: e.message,
              location: e.location,
              expression: e.expression,
            }))
          );
        }

        const problemDetails = extractProblemDetails(error);
        const errorContext = buildErrorContext(error);

        if (problemDetails) {
          toast.error(problemDetails.title || 'Workflow validation failed', {
            description: problemDetails.detail,
            duration: Infinity,
            action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
          });
        } else {
          toast.error('Failed to start workflow test', {
            duration: Infinity,
            action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
          });
        }

        return false;
      }
    },
    [activeWorkflowId, executeWorkflow, setLastRunVersionId, queryClient, setNodeErrors, clearNodeErrors, fixWithAI],
  );

  /**
   * Execute workflow (opens input dialog if needed)
   */
  const handleExecute = useCallback(
    (inputFields: Array<{ id: string; variableName: string; required: boolean }>) => {
      if (!activeWorkflowId) {
        toast.error('Please save the workflow first');
        return;
      }

      // If inputs required, caller should open input dialog
      if (inputFields.length > 0) {
        return { needsInput: true };
      }

      void runWorkflowTest();
      return { needsInput: false };
    },
    [activeWorkflowId, runWorkflowTest],
  );

  /**
   * Execute workflow with provided input data
   */
  const handleExecuteWithInput = useCallback(
    async (
      inputFields: Array<{ id: string; variableName: string; required: boolean }>,
      inputData: Record<string, unknown>,
    ) => {
      if (!activeWorkflowId) {
        toast.error('Please save the workflow first');
        return;
      }

      const transformedInputData: Record<string, unknown> = {};
      inputFields.forEach((field) => {
        transformedInputData[field.variableName] = inputData[field.id];
      });

      try {
        await runWorkflowTest(transformedInputData);
      } catch (error) {
        // Error already handled by runWorkflowTest
      }
    },
    [activeWorkflowId, runWorkflowTest],
  );

  /**
   * Publish workflow
   */
  const handlePublish = useCallback(async () => {
    if (!activeWorkflowId) {
      toast.error('Save the workflow before publishing');
      return;
    }

    // Clear node errors on publish
    clearNodeErrors();

    try {
      await publishWorkflow(activeWorkflowId);
      setLastRunVersionId(null);

      // Check if any form trigger has a live URL to show
      const triggers = useTriggersStore.getState().getWorkflowTriggers(activeWorkflowId);
      const formUrl = triggers
        .filter(t => t.key === 'form.hosted')
        .map(t => (t.ui_meta as Record<string, unknown>)?.form_url as string | undefined)
        .find(Boolean);

      if (formUrl) {
        toast.success('Workflow published', {
          description: `Live form URL: ${formUrl}`,
          duration: 8000,
        });
      } else {
        toast.success('Workflow published', {
          description: 'The workflow is now live',
        });
      }
    } catch (error) {
      console.error('Failed to publish workflow:', error);
      const nodeErrors = extractNodeErrors(error);
      if (nodeErrors.length > 0) {
        setNodeErrors(
          nodeErrors.map((e) => ({
            nodeId: e.node_id,
            code: e.code,
            message: e.message,
            location: e.location,
            expression: e.expression,
          }))
        );
      }
      const problemDetails = extractProblemDetails(error);
      const errorContext = buildErrorContext(error);
      if (problemDetails) {
        toast.error(problemDetails.title || 'Failed to publish workflow', {
          description: problemDetails.detail,
          action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
        });
      } else {
        toast.error('Failed to publish workflow', {
          action: { label: '✦ Fix with AI', onClick: () => fixWithAI(errorContext, activeWorkflowId ?? undefined) },
        });
      }
      throw error;
    }
  }, [activeWorkflowId, publishWorkflow, setLastRunVersionId, clearNodeErrors, setNodeErrors, fixWithAI]);

  /**
   * Restore workflow version
   */
  const handleRestoreVersion = useCallback(
    async (versionId: number) => {
      if (!activeWorkflowId || !loadedWorkflow) {
        toast.error('Select a workflow before restoring a version');
        return;
      }

      try {
        const restored = await restoreWorkflowVersion(activeWorkflowId, {
          versionId,
        });
        useTriggersStore.getState().loadTriggersFromSpec(activeWorkflowId, restored.spec.triggers ?? []);
        setNodes(normalizeNodes(restored.graph.nodes, functionBlocksMap));
        setEdges(normalizeEdges(restored.graph.edges));
        useCanvasStore.getState().markClean();
        setLastRunVersionId(null);
        resetSavedDataRef.current?.();
        toast.success('Version restored', {
          description: `Draft now matches version ${versionId}`,
        });
      } catch (error) {
        console.error('Failed to restore workflow version:', error);
        if (error instanceof BackendAPIError && error.status === 409) {
          await handleDraftConflictCallback();
          return;
        }
        toast.error('Failed to restore version', {
          description: error instanceof BackendAPIError ? error.message : undefined,
        });
      }
    },
    [
      activeWorkflowId,
      loadedWorkflow,
      restoreWorkflowVersion,
      setNodes,
      setEdges,
      functionBlocksMap,
      handleDraftConflictCallback,
      setLastRunVersionId,
    ],
  );

  // ==================== MANAGEMENT ACTIONS ====================

  /**
   * Create new workflow
   */
  const handleNewWorkflow = useCallback(async () => {
    try {
      const workflow = await createWorkflow(generateWorkflowName(), { nodes: [], edges: [] });
      navigate(`/workflows/${workflow.workflow_id}`, { replace: true });
    } catch (error) {
      showError(error, { action: 'create new workflow', componentName: 'useWorkflowActions' });
    }
  }, [createWorkflow, navigate]);

  /**
   * Load workflow
   */
  const handleLoadWorkflow = useCallback(
    (workflow: WorkflowListItem) => {
      navigate(`/workflows/${workflow.workflow_id}`, { replace: true });
    },
    [navigate],
  );

  /**
   * Delete workflow
   */
  const handleDeleteWorkflow = useCallback(
    async (workflowId: string) => {
      if (!confirm('Are you sure you want to delete this workflow?')) return;

      try {
        await deleteWorkflow(workflowId);
        if (activeWorkflowId === workflowId) {
          navigate('/workflows', { replace: true });
        }
      } catch (error) {
        console.error('Failed to delete workflow:', error);
        toast.error('Failed to delete workflow');
      }
    },
    [deleteWorkflow, activeWorkflowId, navigate],
  );

  /**
   * Rename workflow
   */
  const handleRenameWorkflow = useCallback(
    async (workflowId: string, newName: string) => {
      if (!newName.trim()) {
        toast.error('Workflow name cannot be empty');
        return;
      }

      try {
        await updateWorkflowMetadata(workflowId, { name: newName.trim() });
        if (activeWorkflowId === workflowId) {
          setWorkflowName(newName.trim());
        }
        toast.success('Workflow renamed successfully');
      } catch (error) {
        console.error('Failed to rename workflow:', error);
        toast.error('Failed to rename workflow');
        throw error;
      }
    },
    [updateWorkflowMetadata, activeWorkflowId, setWorkflowName],
  );

  /**
   * Export workflow
   */
  const handleExportWorkflow = useCallback(
    async (workflowId: string) => {
      try {
        await exportWorkflow(workflowId);
        toast.success('Workflow exported successfully');
      } catch (error) {
        console.error('Failed to export workflow:', error);
        toast.error('Failed to export workflow');
      }
    },
    [exportWorkflow],
  );

  /**
   * Import workflow
   */
  const handleImportWorkflow = useCallback(
    async (file: File, options: { name?: string; importTriggers: boolean }) => {
      try {
        const result = await importWorkflow(file, options);
        navigate(`/workflows/${result.workflow_id}`, { replace: true });
        toast.success(`Workflow "${result.name}" imported successfully`);
      } catch (error) {
        console.error('Failed to import workflow:', error);
        if (error instanceof BackendAPIError) {
          toast.error(`Failed to import workflow: ${error.message}`);
        } else {
          toast.error('Failed to import workflow');
        }
        throw error;
      }
    },
    [importWorkflow, navigate],
  );

  // ==================== AUTOSAVE ACTION ====================

  /**
   * Autosave callback for debounced saves
   */
  const autosaveCallback = useCallback(
    async (data: { nodes: Node<WorkflowNodeData>[]; edges: WorkflowEdge[] }) => {
      if (!activeWorkflowId || !loadedWorkflow) {
        return;
      }

      setAutosaveStatus('saving');
      try {
        await saveWorkflowDraft(activeWorkflowId, {
          graph: { nodes: data.nodes, edges: data.edges },
        });
        setLastRunVersionId(null);
        setAutosaveStatus('saved');
        // Mark canvas as clean after successful save (unified save system)
        useCanvasStore.getState().markClean();
        setTimeout(() => setAutosaveStatus('idle'), 2000);
      } catch (error) {
        if (error instanceof BackendAPIError && error.status === 409) {
          await handleDraftConflict({
            selectedWorkflowId: activeWorkflowId,
            reloadWorkflow,
            setNodes,
            setEdges,
            functionBlocksMap,
            setLastRunVersionId,
            resetSavedDataRef,
            invalidateWorkflowVersions: () => {
              void queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(activeWorkflowId) });
            },
          });
        } else {
          console.error('Autosave failed:', error);
        }
        setAutosaveStatus('error');
        throw error;
      }
    },
    [
      activeWorkflowId,
      loadedWorkflow,
      saveWorkflowDraft,
      setAutosaveStatus,
      reloadWorkflow,
      setNodes,
      setEdges,
      functionBlocksMap,
      setLastRunVersionId,
      queryClient,
    ],
  );

  // ==================== COMPUTED STATE ====================

  const lifecycleStatus = useMemo(
    () =>
      loadedWorkflow
        ? {
            lastTestedVersionId: lastRunVersionId,
          }
        : null,
    [loadedWorkflow, lastRunVersionId],
  );

  const capabilities = useMemo(() => {
    const canRun = Boolean(activeWorkflowId);
    const runDisabledReason = canRun ? undefined : 'Save the workflow before testing';
    const canPublish = Boolean(activeWorkflowId);
    const publishDisabledReason = canPublish ? undefined : 'Save the workflow before publishing';

    return { canRun, runDisabledReason, canPublish, publishDisabledReason };
  }, [activeWorkflowId]);

  return {
    // Lifecycle actions
    handleSave,
    handleExecute,
    handleExecuteWithInput,
    handlePublish,
    handleRestoreVersion,

    // Management actions
    handleNewWorkflow,
    handleLoadWorkflow,
    handleDeleteWorkflow,
    handleRenameWorkflow,
    handleExportWorkflow,
    handleImportWorkflow,

    // Test with real trigger events
    runWorkflowTest,
    runWorkflowWithTriggerEvent,

    // Autosave
    autosaveCallback,

    // State
    lifecycleStatus,
    lastRunVersionId,
    setLastRunVersionId,
    canRun: capabilities.canRun,
    canPublish: capabilities.canPublish,
    runDisabledReason: capabilities.runDisabledReason,
    publishDisabledReason: capabilities.publishDisabledReason,

    // Refs for autosave coordination
    resetSavedDataRef,
  };
}
