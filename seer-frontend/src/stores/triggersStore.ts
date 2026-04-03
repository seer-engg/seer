import type { StateCreator } from 'zustand';

import type { TriggerSpec } from '@/types/workflow-spec';

import { createStore } from './createStore';
import { useCanvasStore } from './canvasStore';

const markCanvasDirty = () => {
  try {
    useCanvasStore.getState().markDirty();
  } catch {
    // Ignore if canvas store not initialized yet (e.g., during SSR/tests)
  }
};

export interface TriggersStore {
  // Workflow triggers (draft state only)
  workflowTriggers: Map<string, TriggerSpec[]>;

  // Actions
  addTrigger: (workflowId: string, triggerId: string, trigger: Omit<TriggerSpec, 'id'>) => TriggerSpec;
  updateTrigger: (workflowId: string, triggerId: string, updates: Partial<TriggerSpec>) => TriggerSpec | null;
  removeTrigger: (workflowId: string, triggerId: string) => void;
  renameTrigger: (workflowId: string, oldId: string, newId: string) => boolean;
  loadTriggersFromSpec: (workflowId: string, triggers: TriggerSpec[]) => void;
  getWorkflowTriggers: (workflowId: string) => TriggerSpec[];
  clearWorkflowTriggers: (workflowId: string) => void;
}

const createTriggersStore: StateCreator<TriggersStore> = (set, get) => ({
  workflowTriggers: new Map(),
  addTrigger(workflowId, triggerId, trigger) {
    const state = get();
    const triggers = state.workflowTriggers.get(workflowId) || [];
    const nextTrigger: TriggerSpec = { ...trigger, id: triggerId };
    const newTriggers = new Map(state.workflowTriggers);
    newTriggers.set(workflowId, [...triggers, nextTrigger]);
    set({ workflowTriggers: newTriggers });
    markCanvasDirty();
    return nextTrigger;
  },
  updateTrigger(workflowId, triggerId, updates) {
    const state = get();
    const triggers = state.workflowTriggers.get(workflowId) || [];
    let updated: TriggerSpec | null = null;
    const newTriggers = new Map(state.workflowTriggers);
    newTriggers.set(
      workflowId,
      triggers.map((trigger) => {
        if (trigger.id !== triggerId) return trigger;
        updated = { ...trigger, ...updates };
        return updated;
      }),
    );
    set({ workflowTriggers: newTriggers });
    if (updated) {
      markCanvasDirty();
    }
    return updated;
  },
  removeTrigger(workflowId, triggerId) {
    const state = get();
    const triggers = state.workflowTriggers.get(workflowId) || [];
    const newTriggers = new Map(state.workflowTriggers);
    newTriggers.set(
      workflowId,
      triggers.filter((trigger) => trigger.id !== triggerId),
    );
    set({ workflowTriggers: newTriggers });
    markCanvasDirty();
  },
  renameTrigger(workflowId, oldId, newId) {
    const state = get();
    const triggers = state.workflowTriggers.get(workflowId) || [];
    const triggerIndex = triggers.findIndex((t) => t.id === oldId);
    if (triggerIndex === -1) {
      return false;
    }
    const newTriggers = new Map(state.workflowTriggers);
    const updatedTriggers = triggers.map((trigger) =>
      trigger.id === oldId ? { ...trigger, id: newId } : trigger,
    );
    newTriggers.set(workflowId, updatedTriggers);
    set({ workflowTriggers: newTriggers });
    markCanvasDirty();
    return true;
  },
  loadTriggersFromSpec(workflowId, triggers) {
    const state = get();
    const newTriggers = new Map(state.workflowTriggers);
    newTriggers.set(workflowId, triggers);
    set({ workflowTriggers: newTriggers });
  },
  getWorkflowTriggers(workflowId) {
    const state = get();
    return state.workflowTriggers.get(workflowId) || [];
  },
  clearWorkflowTriggers(workflowId) {
    const state = get();
    if (!state.workflowTriggers.has(workflowId)) return;
    const next = new Map(state.workflowTriggers);
    next.delete(workflowId);
    set({ workflowTriggers: next });
  },
});

export const useTriggersStore = createStore(createTriggersStore);
