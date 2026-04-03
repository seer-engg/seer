/**
 * Test utilities for integration and unit tests
 */

import { queryClient } from '@/lib/query-client';
import { useCanvasStore } from '@/stores/canvasStore';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useTriggersStore } from '@/stores/triggersStore';
import { useUIStore } from '@/stores/uiStore';
import { useChatStore } from '@/stores/chatStore';

// Mock localStorage for tests
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
    get length() {
      return Object.keys(store).length;
    },
    key: (index: number) => {
      const keys = Object.keys(store);
      return keys[index] || null;
    },
  };
})();

// Set up localStorage mock globally for all tests
if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
    writable: true,
  });
}

/**
 * Reset all Zustand stores to their initial state
 * Call this in beforeEach() to ensure test isolation
 */
export function resetAllStores() {
  queryClient.clear();

  // Clear localStorage
  if (typeof window !== 'undefined' && window.localStorage) {
    window.localStorage.clear();
  }

  // Reset canvas store
  useCanvasStore.getState().reset();

  // Reset workflow store by setting initial state
  useWorkflowStore.setState({
    error: null,
    isCreating: false,
    isUpdating: false,
    isSavingDraft: false,
    isPublishing: false,
    isDeleting: false,
    isRestoringVersion: false,
    isExecuting: false,
    selectedWorkflowId: null,
    workflowName: '',
    workflowInputData: {},
    isLoadingWorkflow: false,
  });

  // Reset triggers store by setting initial state
  useTriggersStore.setState({
    workflowTriggers: new Map(),
  });

  // Reset UI store by setting initial state
  useUIStore.setState({
    buildChatPanelCollapsed: false,
    activeRightPanelTab: 'build',
    proposalPreview: null,
    lastRunVersionId: null,
    pendingConnection: {
      mode: null,
      sourceNodeId: null,
      targetNodeId: null,
      edgeId: null,
    },
  });

  // Reset chat store by setting initial state
  useChatStore.setState({
    transientMessages: [],
    transientSessionId: null,
    isLoading: false,
  });
}
