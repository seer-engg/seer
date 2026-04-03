import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge, FunctionBlockSchema } from '@/components/workflows/types';
import { toast } from '@/components/ui/sonner';
import { normalizeEdges, normalizeNodes } from '@/lib/workflow-normalization';
import type { WorkflowModel } from '@/lib/workflows-api';
import { useCanvasStore } from '@/stores';
import { useTriggersStore } from '@/stores/triggersStore';

interface UseWorkflowSyncParams {
  urlWorkflowId: string | undefined;
  loadedWorkflow: WorkflowModel | null;
  workflowError: unknown;
  isFetchingWorkflow: boolean;
  isDirty: boolean;
  functionBlocksReady: boolean;
  functionBlocksMap: Map<string, FunctionBlockSchema>;
  setSelectedWorkflowId: (id: string | null) => void;
  setWorkflowName: (name: string) => void;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  setProposalPreview: (preview: unknown) => void;
  setLastRunVersionId: (id: number | null) => void;
  setIsLoadingWorkflow: (loading: boolean) => void;
}

export function useWorkflowSync({
  urlWorkflowId,
  loadedWorkflow,
  workflowError,
  isFetchingWorkflow,
  isDirty,
  functionBlocksReady,
  functionBlocksMap,
  setSelectedWorkflowId,
  setWorkflowName,
  setNodes,
  setEdges,
  setProposalPreview,
  setLastRunVersionId,
  setIsLoadingWorkflow,
}: UseWorkflowSyncParams) {
  const navigate = useNavigate();
  const resetSavedDataRef = useRef<(() => void) | null>(null);
  const hydratedWorkflowStateRef = useRef<{ workflowRevision: string; normalizationRevision: string } | null>(null);
  const functionBlockSignature = Array.from(functionBlocksMap.keys()).sort().join('|');

  useEffect(() => {
    setIsLoadingWorkflow(Boolean(urlWorkflowId) && isFetchingWorkflow && !loadedWorkflow);
  }, [urlWorkflowId, isFetchingWorkflow, loadedWorkflow, setIsLoadingWorkflow]);

  useEffect(() => {
    if (!urlWorkflowId) {
      hydratedWorkflowStateRef.current = null;
      setSelectedWorkflowId(null);
      setWorkflowName('My Workflow');
      setNodes([]);
      setEdges([]);
      setProposalPreview(null);
      setLastRunVersionId(null);
      resetSavedDataRef.current?.();
      return;
    }

    if (workflowError) {
      console.error('Failed to load workflow from query:', workflowError);
      toast.error('Failed to load workflow', {
        description: 'The workflow may not exist or you may not have access to it.',
      });
      navigate('/workflows', { replace: true });
      return;
    }

    if (!loadedWorkflow || loadedWorkflow.workflow_id !== urlWorkflowId) {
      return;
    }

    if (!functionBlocksReady) {
      return;
    }

    const workflowRevision = `${loadedWorkflow.workflow_id}:${loadedWorkflow.updated_at}`;
    const normalizationRevision = `${workflowRevision}:${functionBlockSignature}`;
    const previousWorkflowId = hydratedWorkflowStateRef.current?.workflowRevision.split(':', 1)[0] ?? null;

    if (hydratedWorkflowStateRef.current?.normalizationRevision === normalizationRevision) {
      return;
    }

    if (isDirty && previousWorkflowId === loadedWorkflow.workflow_id) {
      return;
    }

    hydratedWorkflowStateRef.current = {
      workflowRevision,
      normalizationRevision,
    };
    useTriggersStore.getState().loadTriggersFromSpec(
      loadedWorkflow.workflow_id,
      loadedWorkflow.spec.triggers ?? [],
    );
    setSelectedWorkflowId(loadedWorkflow.workflow_id);
    setWorkflowName(loadedWorkflow.name);
    setNodes(normalizeNodes(loadedWorkflow.graph.nodes, functionBlocksMap));
    setEdges(normalizeEdges(loadedWorkflow.graph.edges));
    useCanvasStore.getState().markClean();
    setProposalPreview(null);
    setLastRunVersionId(null);
    resetSavedDataRef.current?.();
  }, [
    urlWorkflowId,
    loadedWorkflow,
    workflowError,
    isDirty,
    functionBlocksReady,
    functionBlocksMap,
    functionBlockSignature,
    navigate,
    setNodes,
    setEdges,
    setProposalPreview,
    setSelectedWorkflowId,
    setWorkflowName,
    setLastRunVersionId,
  ]);

  return { resetSavedDataRef };
}
