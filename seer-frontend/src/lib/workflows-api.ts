import { backendApiClient } from '@/lib/api-client';
import { workflowSpecToGraph } from '@/lib/workflow-graph';
import type {
  WorkflowListResponse,
  WorkflowResponse,
  WorkflowVersionListResponse,
  WorkflowSummary,
} from '@/types/workflow-spec';
import type { WorkflowGraphData } from '@/lib/workflow-graph';

export interface WorkflowModel extends WorkflowResponse {
  graph: WorkflowGraphData;
}

export type WorkflowListItem = WorkflowSummary;

export function toWorkflowModel(response: WorkflowResponse): WorkflowModel {
  return {
    ...response,
    graph: workflowSpecToGraph(response.spec),
  };
}

export async function listWorkflows() {
  const response = await backendApiClient.request<WorkflowListResponse>('/api/v1/workflows', {
    method: 'GET',
  });

  return response.items;
}

export async function getWorkflowDetail(workflowId: string) {
  return backendApiClient.request<WorkflowResponse>(`/api/v1/workflows/${workflowId}`, {
    method: 'GET',
  });
}

export async function cancelWorkflowRun(runId: string) {
  return backendApiClient.request(`/api/v1/runs/${runId}/cancel`, {
    method: 'POST',
  });
}

export async function listWorkflowVersions(workflowId: string) {
  return backendApiClient.request<WorkflowVersionListResponse>(
    `/api/v1/workflows/${workflowId}/versions`,
    { method: 'GET' },
  );
}
