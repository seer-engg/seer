import type { StateCreator } from 'zustand';

import type { ChatMessage, ChatExecutionStatus, StreamActivity } from '@/components/workflows/buildtypes';

import { createStore } from './createStore';

export interface ChatStore {
  transientMessages: ChatMessage[];
  transientSessionId: number | null;
  input: string;
  selectedModel: string;
  isLoading: boolean;
  currentSessionId: number | null;
  currentThreadId: string | null;
  currentExecutionTaskId: string | null;
  sessionExecutionStatus: ChatExecutionStatus | null;
  pendingInterruptData: ChatMessage['interruptData'] | null;
  proposalActionLoading: number | null;
  pendingAutoSendMessage: string | null;

  // Streaming execution state
  executionStatus: ChatExecutionStatus | null;
  /** Accumulated list of tool invocations during the current agent run */
  streamingActivities: StreamActivity[];

  // Streaming state
  isStreaming: boolean;
  streamingContent: string;

  appendTransientMessage: (message: ChatMessage) => void;
  setTransientSessionId: (sessionId: number | null) => void;
  clearTransientMessages: () => void;
  setInput: (value: string) => void;
  setSelectedModel: (model: string) => void;
  setIsLoading: (value: boolean) => void;
  setCurrentSessionId: (sessionId: number | null) => void;
  setCurrentThreadId: (threadId: string | null) => void;
  setCurrentExecutionTaskId: (taskId: string | null) => void;
  setSessionExecutionStatus: (status: ChatExecutionStatus | null) => void;
  setPendingInterruptData: (interruptData: ChatMessage['interruptData'] | null) => void;
  setProposalActionLoading: (proposalId: number | null) => void;
  setPendingAutoSendMessage: (message: string | null) => void;

  // Streaming execution actions
  setExecutionStatus: (status: ChatExecutionStatus | null) => void;
  pushStreamingActivity: (tool: string) => void;
  completeStreamingActivity: (tool: string) => void;
  pushStreamingMessage: (content: string) => void;
  startStreaming: (sessionId: number) => void;
  stopStreaming: () => void;

  // Streaming actions
  setIsStreaming: (value: boolean) => void;
  appendStreamingContent: (token: string) => void;
  clearStreamingContent: () => void;

  resetChatState: () => void;
}

const initialState: Omit<
  ChatStore,
  | 'appendTransientMessage'
  | 'setTransientSessionId'
  | 'clearTransientMessages'
  | 'setInput'
  | 'setSelectedModel'
  | 'setIsLoading'
  | 'setCurrentSessionId'
  | 'setCurrentThreadId'
  | 'setCurrentExecutionTaskId'
  | 'setSessionExecutionStatus'
  | 'setPendingInterruptData'
  | 'setProposalActionLoading'
  | 'setPendingAutoSendMessage'
  | 'setExecutionStatus'
  | 'pushStreamingActivity'
  | 'completeStreamingActivity'
  | 'pushStreamingMessage'
  | 'startStreaming'
  | 'stopStreaming'
  | 'resetChatState'
> = {
  transientMessages: [],
  transientSessionId: null,
  input: '',
  selectedModel: '',
  isLoading: false,
  currentSessionId: null,
  currentThreadId: null,
  currentExecutionTaskId: null,
  sessionExecutionStatus: null,
  pendingInterruptData: null,
  proposalActionLoading: null,
  executionStatus: null,
  streamingActivities: [],
  pendingAutoSendMessage: null,
  isStreaming: false,
  streamingContent: '',
};

const createChatStore: StateCreator<ChatStore> = (set) => ({
  ...initialState,
  appendTransientMessage: (message) =>
    set((state) => ({
      transientMessages: [...state.transientMessages, message],
    })),
  setTransientSessionId: (sessionId) => set({ transientSessionId: sessionId }),
  clearTransientMessages: () => set({ transientMessages: [], transientSessionId: null }),
  setInput: (value) => set({ input: value }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  setIsLoading: (value) => set({ isLoading: value }),
  setCurrentSessionId: (sessionId) => set({ currentSessionId: sessionId }),
  setCurrentThreadId: (threadId) => set({ currentThreadId: threadId }),
  setCurrentExecutionTaskId: (taskId) => set({ currentExecutionTaskId: taskId }),
  setSessionExecutionStatus: (status) => set({ sessionExecutionStatus: status }),
  setPendingInterruptData: (interruptData) => set({ pendingInterruptData: interruptData }),
  setProposalActionLoading: (proposalId) => set({ proposalActionLoading: proposalId }),
  setPendingAutoSendMessage: (message) => set({ pendingAutoSendMessage: message }),

  // Streaming execution actions
  setExecutionStatus: (status) => set({ executionStatus: status }),
  pushStreamingActivity: (tool: string) =>
    set((state) => ({
      streamingActivities: [
        ...state.streamingActivities,
        { type: 'tool' as const, tool, status: 'running' as const },
      ],
    })),
  completeStreamingActivity: (tool: string) =>
    set((state) => {
      const activities = [...state.streamingActivities];
      // Mark the last running tool entry with this name as done
      for (let i = activities.length - 1; i >= 0; i--) {
        const a = activities[i];
        if (a.type === 'tool' && a.tool === tool && a.status === 'running') {
          activities[i] = { ...a, status: 'done' };
          break;
        }
      }
      return { streamingActivities: activities };
    }),
  pushStreamingMessage: (content: string) =>
    set((state) => ({
      streamingActivities: [
        ...state.streamingActivities,
        { type: 'ai_message' as const, content },
      ],
    })),
  startStreaming: (_sessionId) =>
    set({
      executionStatus: 'queued',
      streamingActivities: [],
    }),
  stopStreaming: () =>
    set({
      streamingActivities: [],
      executionStatus: null,
    }),

  setIsStreaming: (value) => set({ isStreaming: value }),
  appendStreamingContent: (token) => set((state) => ({ streamingContent: state.streamingContent + token })),
  clearStreamingContent: () => set({ streamingContent: '' }),

  resetChatState: () => set(() => ({ ...initialState })),
});

export const useChatStore = createStore(createChatStore);
