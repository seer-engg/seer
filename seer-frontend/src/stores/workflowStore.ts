import type { StateCreator } from 'zustand';

import { backendApiClient } from '@/lib/api-client';
import { queryClient } from '@/lib/query-client';
import { workflowKeys } from '@/lib/query-keys';
import type { WorkflowGraphData } from '@/lib/workflow-graph';
import { graphToWorkflowSpec } from '@/lib/workflow-graph';
import {
  getWorkflowDetail,
  toWorkflowModel,
  type WorkflowListItem,
  type WorkflowModel,
} from '@/lib/workflows-api';
import type {
  JsonValue,
  RunResponse,
  WorkflowDraftPatchRequest,
  WorkflowResponse,
  WorkflowSpec,
  WorkflowVersionRestoreRequest,
} from '@/types/workflow-spec';

import { createStore } from './createStore';
import { useTriggersStore } from './triggersStore';

export interface WorkflowStore {
  error: string | null;
  isCreating: boolean;
  isUpdating: boolean;
  isSavingDraft: boolean;
  isPublishing: boolean;
  isDeleting: boolean;
  isRestoringVersion: boolean;
  isExecuting: boolean;
  // Phase 1: Consolidated workflow UI state (previously in Workflows.tsx local state)
  selectedWorkflowId: string | null;
  workflowName: string;
  workflowInputData: Record<string, unknown>;
  isLoadingWorkflow: boolean;
  // Phase 1: Setter actions for consolidated state
  setSelectedWorkflowId: (id: string | null) => void;
  setWorkflowName: (name: string) => void;
  setWorkflowInputData: (data: Record<string, unknown>) => void;
  clearWorkflowInputData: () => void;
  setIsLoadingWorkflow: (isLoading: boolean) => void;
  createWorkflow: (
    name: string,
    graph: WorkflowGraphData,
  ) => Promise<WorkflowModel>;
  updateWorkflowMetadata: (workflowId: string, updates: { name?: string }) => Promise<WorkflowModel>;
  saveWorkflowDraft: (
    workflowId: string,
    params: { graph: WorkflowGraphData },
  ) => Promise<WorkflowModel>;
  publishWorkflow: (workflowId: string) => Promise<WorkflowModel>;
  deleteWorkflow: (workflowId: string) => Promise<void>;
  restoreWorkflowVersion: (
    workflowId: string,
    params: { versionId: number },
  ) => Promise<WorkflowModel>;
  executeWorkflow: (
    workflowId: string,
    inputs?: Record<string, JsonValue>,
    config?: Record<string, JsonValue>,
    triggerOverride?: { triggerId: string; envelope: Record<string, unknown> },
  ) => Promise<RunResponse>;
  duplicateWorkflow: (workflowId: string) => Promise<WorkflowModel>;
  exportWorkflow: (workflowId: string) => Promise<void>;
  importWorkflow: (file: File, options: { name?: string; importTriggers: boolean }) => Promise<WorkflowModel>;
};

const toListItem = (workflow: WorkflowModel): WorkflowListItem => ({
  workflow_id: workflow.workflow_id,
  name: workflow.name,
  created_at: workflow.created_at,
  updated_at: workflow.updated_at,
});

const updateWorkflowList = (workflows: WorkflowListItem[], updated: WorkflowModel) => {
  const idx = workflows.findIndex((workflow) => workflow.workflow_id === updated.workflow_id);
  if (idx === -1) {
    return [toListItem(updated), ...workflows];
  }
  const next = [...workflows];
  next[idx] = toListItem(updated);
  return next;
};

const removeWorkflowFromList = (workflows: WorkflowListItem[], workflowId: string) =>
  workflows.filter((workflow) => workflow.workflow_id !== workflowId);

const updateWorkflowListCache = (workflow: WorkflowModel) => {
  queryClient.setQueryData<WorkflowListItem[]>(workflowKeys.list(), (current = []) =>
    updateWorkflowList(current, workflow),
  );
};

interface BackendClientWithMeta {
  getBaseUrl?: () => string;
  baseUrl?: string;
  getToken?: () => Promise<string | null>;
}

const createWorkflowImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  name: string,
  graph: WorkflowGraphData,
  existingSpec?: WorkflowSpec,
) => {
  set({ isCreating: true, error: null });
  try {
    const spec = graphToWorkflowSpec(graph, existingSpec);
    const response = await backendApiClient.request<WorkflowResponse>('/api/v1/workflows', {
      method: 'POST',
      body: { name, spec },
    });
    const workflow = toWorkflowModel(response);
    queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
    updateWorkflowListCache(workflow);
    set({ isCreating: false });
    return workflow;
  } catch (error) {
    set({ isCreating: false, error: error instanceof Error ? error.message : 'Failed to create workflow' });
    throw error;
  }
};

const updateWorkflowMetadataImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
  updates: { name?: string },
) => {
  set({ isUpdating: true, error: null });
  try {
    const response = await backendApiClient.request<WorkflowResponse>(`/api/v1/workflows/${workflowId}`, {
      method: 'PUT',
      body: updates,
    });
    const workflow = toWorkflowModel(response);
    queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
    updateWorkflowListCache(workflow);
    set({ isUpdating: false });
    return workflow;
  } catch (error) {
    set({ isUpdating: false, error: error instanceof Error ? error.message : 'Failed to update workflow' });
    throw error;
  }
};

const saveWorkflowDraftImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
  graph: WorkflowGraphData,
) => {
  set({ isSavingDraft: true, error: null });
  try {
    const existing =
      queryClient.getQueryData<WorkflowModel>(workflowKeys.detail(workflowId)) ??
      (await queryClient.fetchQuery({
        queryKey: workflowKeys.detail(workflowId),
        staleTime: 0,
        queryFn: async () => toWorkflowModel(await getWorkflowDetail(workflowId)),
      }));
    // Get triggers from the triggers store
    const triggers = useTriggersStore.getState().getWorkflowTriggers(workflowId);

    // Include triggers in the spec (no transformation needed)
    const spec = graphToWorkflowSpec(graph, existing?.spec, triggers);
    const body: WorkflowDraftPatchRequest = { spec };
    const response = await backendApiClient.request<WorkflowResponse>(`/api/v1/workflows/${workflowId}/draft`, {
      method: 'PATCH',
      body: body as unknown as Record<string, unknown>,
    });
    const workflow = toWorkflowModel(response);
    queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
    updateWorkflowListCache(workflow);
    set({ isSavingDraft: false });
    return workflow;
  } catch (error) {
    set({ isSavingDraft: false, error: error instanceof Error ? error.message : 'Failed to save draft' });
    throw error;
  }
};

const publishWorkflowImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
) => {
  set({ isPublishing: true, error: null });
  try {
    const response = await backendApiClient.request<WorkflowResponse>(`/api/v1/workflows/${workflowId}/publish`, {
      method: 'POST',
    });
    const workflow = toWorkflowModel(response);
    // Load triggers from spec into the triggers store (includes enriched ui_meta from publish)
    useTriggersStore.getState().loadTriggersFromSpec(workflowId, response.spec.triggers || []);
    queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
    updateWorkflowListCache(workflow);
    set({ isPublishing: false });
    void queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(workflowId) });
    return workflow;
  } catch (error) {
    set({ isPublishing: false, error: error instanceof Error ? error.message : 'Failed to publish workflow' });
    throw error;
  }
};

const deleteWorkflowImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
) => {
  set({ isDeleting: true, error: null });
  try {
    await backendApiClient.request(`/api/v1/workflows/${workflowId}`, { method: 'DELETE' });
    queryClient.setQueryData<WorkflowListItem[]>(workflowKeys.list(), (current = []) =>
      removeWorkflowFromList(current, workflowId),
    );
    queryClient.removeQueries({ queryKey: workflowKeys.detail(workflowId) });
    queryClient.removeQueries({ queryKey: workflowKeys.versionList(workflowId) });
    set({ isDeleting: false });
  } catch (error) {
    set({ isDeleting: false, error: error instanceof Error ? error.message : 'Failed to delete workflow' });
    throw error;
  }
};

const restoreWorkflowVersionImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
  versionId: number,
) => {
  set({ isRestoringVersion: true, error: null });
  try {
    const body: WorkflowVersionRestoreRequest = {};
    const response = await backendApiClient.request<WorkflowResponse>(
      `/api/v1/workflows/${workflowId}/versions/${versionId}/restore`,
      { method: 'POST', body: body as unknown as Record<string, unknown> },
    );
    const workflow = toWorkflowModel(response);
    useTriggersStore.getState().loadTriggersFromSpec(workflowId, response.spec.triggers || []);
    queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
    updateWorkflowListCache(workflow);
    set({ isRestoringVersion: false });
    void queryClient.invalidateQueries({ queryKey: workflowKeys.versionList(workflowId) });
    return workflow;
  } catch (error) {
    set({ isRestoringVersion: false, error: error instanceof Error ? error.message : 'Failed to restore workflow version' });
    throw error;
  }
};

const executeWorkflowImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  workflowId: string,
  inputs?: Record<string, JsonValue>,
  config?: Record<string, JsonValue>,
  triggerOverride?: { triggerId: string; envelope: Record<string, unknown> },
) => {
  set({ isExecuting: true, error: null });
  try {
    const payload: Record<string, unknown> = {
      inputs: inputs ?? {},
      config: config ?? {},
    };

    // Add trigger override if provided (for testing with real trigger events)
    if (triggerOverride) {
      payload.trigger_id = triggerOverride.triggerId;
      payload.trigger_event_override = triggerOverride.envelope;
    }

    const response = await backendApiClient.request<RunResponse>(`/api/v1/workflows/${workflowId}/runs`, {
      method: 'POST',
      body: payload,
    });
    set({ isExecuting: false });
    return response;
  } catch (error) {
    set({ isExecuting: false, error: error instanceof Error ? error.message : 'Failed to execute workflow' });
    throw error;
  }
};

const exportWorkflowImpl = async (workflowId: string) => {
  const clientMeta = backendApiClient as unknown as BackendClientWithMeta;
  const baseUrl = typeof clientMeta.getBaseUrl === 'function' ? clientMeta.getBaseUrl() : clientMeta.baseUrl ?? '';
  const url = `${baseUrl}/api/v1/workflows/${workflowId}/export`;
  const token = await clientMeta.getToken?.();
  const headers: HeadersInit = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  const response = await fetch(url, { method: 'GET', headers });
  if (!response.ok) throw new Error('Failed to export workflow');
  const contentDisposition = response.headers.get('Content-Disposition');
  const filenameMatch = contentDisposition?.match(/filename="(.+)"/);
  const filename = filenameMatch?.[1] || `workflow-${workflowId}.seer.json`;
  const blob = await response.blob();
  const downloadUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = downloadUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.URL.revokeObjectURL(downloadUrl);
};

const importWorkflowImpl = async (
  set: (partial: Partial<WorkflowStore>) => void,
  file: File,
  options: { name?: string; importTriggers: boolean },
) => {
  set({ error: null });
  const text = await file.text();
  const importData = JSON.parse(text);
  const response = await backendApiClient.request<WorkflowResponse>('/api/v1/workflows/import', {
    method: 'POST',
    body: { import_data: importData, name: options.name, import_triggers: options.importTriggers },
  });
  const workflow = toWorkflowModel(response);
  queryClient.setQueryData(workflowKeys.detail(workflow.workflow_id), workflow);
  updateWorkflowListCache(workflow);
  return workflow;
};

const createWorkflowStore: StateCreator<WorkflowStore> = (set) => ({
  error: null,
  isCreating: false,
  isUpdating: false,
  isSavingDraft: false,
  isPublishing: false,
  isDeleting: false,
  isRestoringVersion: false,
  isExecuting: false,
  selectedWorkflowId: null,
  workflowName: 'My Workflow',
  workflowInputData: {},
  isLoadingWorkflow: false,
  setSelectedWorkflowId: (id) =>
    set((state) => {
      const previousId = state.selectedWorkflowId;
      if (previousId && previousId !== id) {
        useTriggersStore.getState().clearWorkflowTriggers(previousId);
      }
      return { selectedWorkflowId: id };
    }),
  setWorkflowName: (name) => set({ workflowName: name }),
  setWorkflowInputData: (data) => set({ workflowInputData: data }),
  clearWorkflowInputData: () => set({ workflowInputData: {} }),
  setIsLoadingWorkflow: (isLoading) => set({ isLoadingWorkflow: isLoading }),
  async createWorkflow(name, graph) {
    return createWorkflowImpl(set, name, graph);
  },
  async updateWorkflowMetadata(workflowId, updates) {
    return updateWorkflowMetadataImpl(set, workflowId, updates);
  },
  async saveWorkflowDraft(workflowId, { graph }) {
    return saveWorkflowDraftImpl(set, workflowId, graph);
  },
  async publishWorkflow(workflowId) {
    return publishWorkflowImpl(set, workflowId);
  },
  async deleteWorkflow(workflowId) {
    return deleteWorkflowImpl(set, workflowId);
  },
  async restoreWorkflowVersion(workflowId, { versionId }) {
    return restoreWorkflowVersionImpl(set, workflowId, versionId);
  },
  async executeWorkflow(workflowId, inputs, config, triggerOverride) {
    return executeWorkflowImpl(set, workflowId, inputs, config, triggerOverride);
  },
  async duplicateWorkflow(workflowId) {
    const source = await queryClient.fetchQuery({
      queryKey: workflowKeys.detail(workflowId),
      staleTime: 0,
      queryFn: async () => toWorkflowModel(await getWorkflowDetail(workflowId)),
    });
    return createWorkflowImpl(set, `${source.name} (copy)`, source.graph, source.spec);
  },
  async exportWorkflow(workflowId) {
    return exportWorkflowImpl(workflowId);
  },
  async importWorkflow(file, options) {
    return importWorkflowImpl(set, file, options);
  },
});

export const useWorkflowStore = createStore(createWorkflowStore);
