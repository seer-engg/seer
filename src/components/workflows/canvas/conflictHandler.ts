import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import { toast } from '@/components/ui/sonner';
import { normalizeNodes, normalizeEdges } from '@/lib/workflow-normalization';
import type { WorkflowModel } from '@/lib/workflows-api';

export async function handleDraftConflict(params: {
  selectedWorkflowId: string | null;
  reloadWorkflow: (id: string) => Promise<WorkflowModel>;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  setLastRunVersionId: (id: number | null) => void;
  functionBlocksMap: Map<string, unknown>;
  invalidateWorkflowVersions: () => void;
  resetSavedDataRef: React.MutableRefObject<(() => void) | undefined>;
}) {
  if (!params.selectedWorkflowId) return;

  try {
    const latest = await params.reloadWorkflow(params.selectedWorkflowId);
    params.setNodes(normalizeNodes(latest.graph.nodes, params.functionBlocksMap));
    params.setEdges(normalizeEdges(latest.graph.edges));
    params.setLastRunVersionId(null);
    params.resetSavedDataRef.current?.();
    params.invalidateWorkflowVersions();
    toast.error('Draft conflict detected', {
      description: 'Reloaded the latest draft from the server. Please retry your change.',
    });
  } catch (error) {
    console.error('Failed to refresh workflow after conflict:', error);
    toast.error('Draft conflict detected', {
      description: 'Reload failed. Please refresh the page to continue.',
    });
  }
}
