import { create } from 'zustand';
import type { GlobalVariableItem } from '@/lib/api-client';
import {
  listGlobalVariables,
  createGlobalVariable,
  updateGlobalVariable,
  deleteGlobalVariable,
} from '@/lib/api-client';

interface VariablesState {
  variables: GlobalVariableItem[];
  isLoading: boolean;
  error: string | null;

  fetchVariables: () => Promise<void>;
  addVariable: (data: { key: string; value: string; is_secret?: boolean; description?: string }) => Promise<void>;
  editVariable: (id: number, data: { value?: string; is_secret?: boolean; description?: string }) => Promise<void>;
  removeVariable: (id: number) => Promise<void>;
}

export const useVariablesStore = create<VariablesState>()((set, get) => ({
  variables: [],
  isLoading: false,
  error: null,

  fetchVariables: async () => {
    set({ isLoading: true, error: null });
    try {
      const res = await listGlobalVariables();
      set({ variables: res.items, isLoading: false });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'Failed to load variables', isLoading: false });
    }
  },

  addVariable: async (data) => {
    const item = await createGlobalVariable(data);
    set((s) => ({ variables: [...s.variables, item].sort((a, b) => a.key.localeCompare(b.key)) }));
  },

  editVariable: async (id, data) => {
    const item = await updateGlobalVariable(id, data);
    set((s) => ({ variables: s.variables.map((v) => (v.id === id ? item : v)) }));
  },

  removeVariable: async (id) => {
    await deleteGlobalVariable(id);
    set((s) => ({ variables: s.variables.filter((v) => v.id !== id) }));
  },
}));
