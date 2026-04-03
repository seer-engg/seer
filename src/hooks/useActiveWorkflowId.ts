import { useParams } from 'react-router-dom';

import { useWorkflowStore } from '@/stores/workflowStore';

export function useActiveWorkflowId() {
  const { workflowId } = useParams<{ workflowId?: string }>();
  const selectedWorkflowId = useWorkflowStore((state) => state.selectedWorkflowId);

  return workflowId ?? selectedWorkflowId ?? null;
}
