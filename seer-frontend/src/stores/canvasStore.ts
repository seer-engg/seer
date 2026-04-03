import { Node } from '@xyflow/react';
import type { StateCreator } from 'zustand';
import type { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';
import { validateNodeId, renameNodeInGraph } from '@/lib/workflow-rename';
import { createStore } from './createStore';

export type AutosaveStatus = 'idle' | 'saving' | 'saved' | 'error';

/**
 * Node error state for highlighting validation errors on workflow nodes
 */
export interface NodeErrorState {
  nodeId: string;
  code: string;
  message: string;
  location?: string;
  expression?: string;
}

type NodesUpdater =
  | Node<WorkflowNodeData>[]
  | ((prev: Node<WorkflowNodeData>[]) => Node<WorkflowNodeData>[]);
type EdgesUpdater =
  | WorkflowEdge[]
  | ((prev: WorkflowEdge[]) => WorkflowEdge[]);

export interface CanvasStore {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
  selectedNodeId: string | null;
  editingNodeId: string | null;
  autosaveStatus: AutosaveStatus;
  isDirty: boolean;
  lastSavedAt: number | null;
  nodeErrors: Map<string, NodeErrorState>;
  setNodes: (updater: NodesUpdater) => void;
  setEdges: (updater: EdgesUpdater) => void;
  setGraph: (payload: { nodes?: Node<WorkflowNodeData>[]; edges?: WorkflowEdge[] }) => void;
  addNode: (node: Node<WorkflowNodeData>) => void;
  updateNode: (nodeId: string, updates: Partial<WorkflowNodeData>) => void;
  deleteNode: (nodeId: string) => void;
  addEdge: (edge: WorkflowEdge) => void;
  deleteEdge: (edgeId: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  setEditingNodeId: (nodeId: string | null) => void;
  setAutosaveStatus: (status: AutosaveStatus) => void;
  setNodeErrors: (errors: NodeErrorState[]) => void;
  clearNodeErrors: () => void;
  markDirty: () => void;
  markClean: () => void;
  renameNode: (oldId: string, newId: string) => { success: boolean; error?: string };
  reset: () => void;
}

const initialState: Pick<
  CanvasStore,
  'nodes' | 'edges' | 'selectedNodeId' | 'editingNodeId' | 'autosaveStatus' | 'isDirty' | 'lastSavedAt' | 'nodeErrors'
> = {
  nodes: [],
  edges: [],
  selectedNodeId: null,
  editingNodeId: null,
  autosaveStatus: 'idle',
  isDirty: false,
  lastSavedAt: null,
  nodeErrors: new Map<string, NodeErrorState>(),
};

/**
 * Renames a node in the workflow, updating all references atomically.
 * Extracted to satisfy max-lines-per-function lint rule.
 */
function performNodeRename(
  oldId: string,
  newId: string,
  get: () => CanvasStore,
  set: (partial: Partial<CanvasStore> | ((state: CanvasStore) => Partial<CanvasStore>)) => void,
): { success: boolean; error?: string } {
  const state = get();
  const existingIds = state.nodes.map((n) => n.id);

  const validation = validateNodeId(newId, existingIds, oldId);
  if (!validation.valid) {
    return { success: false, error: validation.error };
  }

  const { nodes: updatedNodes, edges: updatedEdges } = renameNodeInGraph(
    oldId,
    newId,
    state.nodes,
    state.edges,
  );

  set((s) => ({
    nodes: updatedNodes,
    edges: updatedEdges,
    selectedNodeId: s.selectedNodeId === oldId ? newId : s.selectedNodeId,
    editingNodeId: s.editingNodeId === oldId ? newId : s.editingNodeId,
    isDirty: true,
  }));

  return { success: true };
}

const createCanvasStore: StateCreator<CanvasStore> = (set, get) => ({
  ...initialState,
  setNodes: (updater) =>
    set((state) => ({
      nodes:
        typeof updater === 'function'
          ? (updater as (prev: Node<WorkflowNodeData>[]) => Node<WorkflowNodeData>[])(state.nodes)
          : updater,
    })),
  setEdges: (updater) =>
    set((state) => ({
      edges:
        typeof updater === 'function'
          ? (updater as (prev: WorkflowEdge[]) => WorkflowEdge[])(state.edges)
          : updater,
    })),
  setGraph: ({ nodes, edges }) =>
    set((state) => ({
      nodes: nodes ?? state.nodes,
      edges: edges ?? state.edges,
      isDirty: false, // Loading a workflow from server marks it as clean
      lastSavedAt: Date.now(),
    })),
  addNode: (node) =>
    set((state) => ({
      nodes: [...state.nodes, node],
    })),
  updateNode: (nodeId, updates) =>
    set((state) => {
      // Clear error for this node when it's edited
      const nextErrors = new Map(state.nodeErrors);
      nextErrors.delete(nodeId);

      return {
        nodeErrors: nextErrors,
        nodes: state.nodes.map((node) => {
          if (node.id !== nodeId) {
            return node;
          }

          const mergedData: WorkflowNodeData = {
            ...node.data,
            ...updates,
          };
          if (updates.config) {
            const originalConfig = node.data?.config || {};
            const mergedConfig = {
              ...originalConfig,
              ...updates.config,
            };

            // Deep merge params to preserve existing param fields when updating
            // Handle all cases: both have params, only original has params, or only update has params
            if ('params' in updates.config || 'params' in originalConfig) {
              const originalParams = originalConfig.params && typeof originalConfig.params === 'object'
                ? originalConfig.params as Record<string, unknown>
                : {};
              const updateParams = updates.config.params && typeof updates.config.params === 'object'
                ? updates.config.params as Record<string, unknown>
                : {};

              // Merge params: update params override original params
              mergedConfig.params = {
                ...originalParams,
                ...updateParams,
              };
            }

            // Always replace fields array if provided (even if empty)
            if ('fields' in updates.config) {
              mergedConfig.fields = updates.config.fields;
            }

            mergedData.config = mergedConfig;
          }
          return {
            ...node,
            data: mergedData,
          };
        }),
      };
    }),
  deleteNode: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.filter((node) => node.id !== nodeId),
      edges: state.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId),
      selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
      editingNodeId: state.editingNodeId === nodeId ? null : state.editingNodeId,
    })),
  addEdge: (edge) =>
    set((state) => ({
      edges: [...state.edges, edge],
    })),
  deleteEdge: (edgeId) =>
    set((state) => ({
      edges: state.edges.filter((edge) => edge.id !== edgeId),
    })),
  setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId }),
  setEditingNodeId: (nodeId) => set({ editingNodeId: nodeId }),
  setAutosaveStatus: (status) => set({ autosaveStatus: status }),
  setNodeErrors: (errors) =>
    set({ nodeErrors: new Map(errors.map((e) => [e.nodeId, e])) }),
  clearNodeErrors: () => set({ nodeErrors: new Map() }),
  markDirty: () => set({ isDirty: true }),
  markClean: () => set({ isDirty: false, lastSavedAt: Date.now() }),
  renameNode: (oldId, newId) => performNodeRename(oldId, newId, get, set),
  reset: () => set(() => ({ ...initialState, nodeErrors: new Map() })),
});

export const useCanvasStore = createStore(createCanvasStore);


