import { createJSONStorage, persist } from 'zustand/middleware';
import type { WorkflowProposalPreview } from '@/components/workflows/buildtypes';
import type { WorkflowCreationMode } from '@/types/discovery';
import { createStore } from './createStore';

const storage = createJSONStorage(() => ({
  getItem: (key: string) => {
    if (typeof window === 'undefined') {
      return null;
    }
    return window.localStorage.getItem(key);
  },
  setItem: (key: string, value: string) => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.setItem(key, value);
  },
  removeItem: (key: string) => {
    if (typeof window === 'undefined') {
      return;
    }
    window.localStorage.removeItem(key);
  },
}));

export interface PendingConnection {
  mode: 'append' | 'insert' | null;
  sourceNodeId: string | null;
  targetNodeId: string | null;
  edgeId: string | null;
  branch?: 'true' | 'false' | 'loop' | 'exit';
}

export interface InlineBlockPickerState {
  visible: boolean;
  sourceNodeId: string;
  position: { x: number; y: number };
}

export interface UIStore {
  isConfigDialogOpen: boolean;
  isImportDialogOpen: boolean;
  isKeymapOpen: boolean;
  isInputDialogOpen: boolean;
  isPaymentModalOpen: boolean;
  isRunOptionsDialogOpen: boolean;
  isTriggerEventPickerOpen: boolean;
  inlineBlockPicker: InlineBlockPickerState | null;
  /** Right contextual pane mode - null means closed */
  rightPaneMode: 'config' | 'executions' | null;
  /** Selected trigger for event picker (when browsing real events) */
  selectedTriggerForPicker: {
    triggerId: string;
    triggerKey: string;
    provider: string;
    connectionId?: number;
    subscriptionId?: number;
    filterParams?: Record<string, unknown>;
  } | null;
  buildChatPanelCollapsed: boolean;
  activeRightPanelTab: 'build' | 'chat' | 'executions';
  proposalPreview: WorkflowProposalPreview | null;
  // Phase 1: Workflow execution state (previously in useWorkflowLifecycle hook)
  lastRunVersionId: number | null;
  // Pending connection for edge interactions
  pendingConnection: PendingConnection;
  handleOrientation: 'vertical' | 'horizontal';
  defaultWorkflowCreationMode: WorkflowCreationMode;
  openConfigDialog: () => void;
  closeConfigDialog: () => void;
  setConfigDialogOpen: (open: boolean) => void;
  setImportDialogOpen: (open: boolean) => void;
  setKeymapOpen: (open: boolean) => void;
  setInputDialogOpen: (open: boolean) => void;
  setPaymentModalOpen: (open: boolean) => void;
  setRunOptionsDialogOpen: (open: boolean) => void;
  setTriggerEventPickerOpen: (open: boolean) => void;
  setSelectedTriggerForPicker: (trigger: {
    triggerId: string;
    triggerKey: string;
    provider: string;
    connectionId?: number;
    subscriptionId?: number;
    filterParams?: Record<string, unknown>;
  } | null) => void;
  toggleBuildChatPanel: () => void;
  setBuildChatPanelCollapsed: (collapsed: boolean) => void;
  setActiveRightPanelTab: (tab: 'build' | 'chat' | 'executions') => void;
  setInlineBlockPicker: (state: InlineBlockPickerState) => void;
  clearInlineBlockPicker: () => void;
  setProposalPreview: (preview: WorkflowProposalPreview | null) => void;
  setRightPaneMode: (mode: 'config' | 'executions' | null) => void;
  closeRightPane: () => void;
  // Phase 1: Setter for execution state
  setLastRunVersionId: (id: number | null) => void;
  setPendingConnection: (connection: PendingConnection) => void;
  clearPendingConnection: () => void;
  setHandleOrientation: (orientation: 'vertical' | 'horizontal') => void;
  toggleHandleOrientation: () => void;
  setDefaultWorkflowCreationMode: (mode: WorkflowCreationMode) => void;
  resetUIState: () => void;
}

const initialState: Pick<
  UIStore,
  | 'isConfigDialogOpen'
  | 'isImportDialogOpen'
  | 'isKeymapOpen'
  | 'isInputDialogOpen'
  | 'isPaymentModalOpen'
  | 'isRunOptionsDialogOpen'
  | 'isTriggerEventPickerOpen'
  | 'selectedTriggerForPicker'
  | 'buildChatPanelCollapsed'
  | 'activeRightPanelTab'
  | 'proposalPreview'
  | 'lastRunVersionId'
  | 'pendingConnection'
  | 'handleOrientation'
  | 'defaultWorkflowCreationMode'
  | 'inlineBlockPicker'
  | 'rightPaneMode'
> = {
  isConfigDialogOpen: false,
  isImportDialogOpen: false,
  isKeymapOpen: false,
  isInputDialogOpen: false,
  isPaymentModalOpen: false,
  isRunOptionsDialogOpen: false,
  isTriggerEventPickerOpen: false,
  selectedTriggerForPicker: null,
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
  handleOrientation: 'vertical',
  defaultWorkflowCreationMode: 'ASK_FIRST',
  inlineBlockPicker: null,
  rightPaneMode: null,
};

export const useUIStore = createStore(
  persist<UIStore>(
    (set) => ({
      ...initialState,
      openConfigDialog: () => set({ isConfigDialogOpen: true }),
      closeConfigDialog: () => set({ isConfigDialogOpen: false }),
      setConfigDialogOpen: (open) => set({ isConfigDialogOpen: open }),
      setImportDialogOpen: (open) => set({ isImportDialogOpen: open }),
      setKeymapOpen: (open) => set({ isKeymapOpen: open }),
      setInputDialogOpen: (open) => set({ isInputDialogOpen: open }),
      setPaymentModalOpen: (open) => set({ isPaymentModalOpen: open }),
      setRunOptionsDialogOpen: (open) => set({ isRunOptionsDialogOpen: open }),
      setTriggerEventPickerOpen: (open) => set({ isTriggerEventPickerOpen: open }),
      setSelectedTriggerForPicker: (trigger) => set({ selectedTriggerForPicker: trigger }),
      toggleBuildChatPanel: () =>
        set((state) => ({
          buildChatPanelCollapsed: !state.buildChatPanelCollapsed,
        })),
      setBuildChatPanelCollapsed: (collapsed) => set({ buildChatPanelCollapsed: collapsed }),
      setActiveRightPanelTab: (tab) => set({ activeRightPanelTab: tab }),
      setInlineBlockPicker: (pickerState) => set({ inlineBlockPicker: pickerState }),
      clearInlineBlockPicker: () => set({ inlineBlockPicker: null }),
      setProposalPreview: (preview) => set({ proposalPreview: preview }),
      setRightPaneMode: (mode) => set({ rightPaneMode: mode }),
      closeRightPane: () => set({ rightPaneMode: null }),
      // Phase 1: Implementation of execution state setter
      setLastRunVersionId: (id) => set({ lastRunVersionId: id }),
      setPendingConnection: (connection) => set({ pendingConnection: connection }),
      clearPendingConnection: () =>
        set({
          pendingConnection: {
            mode: null,
            sourceNodeId: null,
            targetNodeId: null,
            edgeId: null,
          },
        }),
      setHandleOrientation: (orientation) => set({ handleOrientation: orientation }),
      toggleHandleOrientation: () =>
        set((state) => ({
          handleOrientation: state.handleOrientation === 'vertical' ? 'horizontal' : 'vertical',
        })),
      setDefaultWorkflowCreationMode: (mode) => set({ defaultWorkflowCreationMode: mode }),
      resetUIState: () => set({ ...initialState }),
    }),
    {
      name: 'seer-ui-store',
      storage,
      partialize: (state) => ({
        buildChatPanelCollapsed: state.buildChatPanelCollapsed,
        isKeymapOpen: state.isKeymapOpen,
        handleOrientation: state.handleOrientation,
        defaultWorkflowCreationMode: state.defaultWorkflowCreationMode,
      }),
    },
  ),
);
