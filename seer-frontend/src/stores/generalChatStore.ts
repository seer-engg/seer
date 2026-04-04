import type { StateCreator } from 'zustand';
import { createStore } from './createStore';

export interface GeneralChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  model?: string;
  image_urls?: string[];
  thinking?: string[];
  created_at: string;
}

export interface GeneralChatSession {
  id: number;
  title: string | null;
  created_at: string;
  updated_at: string;
  current_execution_status: string | null;
}

export interface AttachedFile {
  name: string;
  content: string;
}

export interface GeneralChatStore {
  sessions: GeneralChatSession[];
  activeSessionId: number | null;
  messages: GeneralChatMessage[];
  input: string;
  selectedModel: string;
  selectedImageModel: string;
  isLoading: boolean;
  executionStatus: string | null;
  pollingEnabled: boolean;
  generateImage: boolean;
  attachedFiles: AttachedFile[];
  streamingContent: string;
  isStreaming: boolean;

  setSessions: (sessions: GeneralChatSession[]) => void;
  setActiveSessionId: (id: number | null) => void;
  setMessages: (messages: GeneralChatMessage[]) => void;
  addMessage: (message: GeneralChatMessage) => void;
  setInput: (value: string) => void;
  setSelectedModel: (model: string) => void;
  setSelectedImageModel: (model: string) => void;
  setIsLoading: (value: boolean) => void;
  setExecutionStatus: (status: string | null) => void;
  setPollingEnabled: (enabled: boolean) => void;
  setGenerateImage: (value: boolean) => void;
  addAttachedFile: (file: AttachedFile) => void;
  removeAttachedFile: (name: string) => void;
  clearAttachedFiles: () => void;
  appendStreamingContent: (token: string) => void;
  finalizeStreaming: (content: string) => void;
  startPolling: () => void;
  stopPolling: () => void;
  reset: () => void;
}

const initialState = {
  sessions: [] as GeneralChatSession[],
  activeSessionId: null as number | null,
  messages: [] as GeneralChatMessage[],
  input: '',
  selectedModel: '',
  selectedImageModel: '',
  isLoading: false,
  executionStatus: null as string | null,
  pollingEnabled: false,
  generateImage: false,
  attachedFiles: [] as AttachedFile[],
  streamingContent: '',
  isStreaming: false,
};

const createGeneralChatStore: StateCreator<GeneralChatStore> = (set) => ({
  ...initialState,
  setSessions: (sessions) => set({ sessions }),
  setActiveSessionId: (id) => set({ activeSessionId: id }),
  setMessages: (messages) => set({ messages }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  setInput: (value) => set({ input: value }),
  setSelectedModel: (model) => set({ selectedModel: model }),
  setSelectedImageModel: (model) => set({ selectedImageModel: model }),
  setIsLoading: (value) => set({ isLoading: value }),
  setExecutionStatus: (status) => set({ executionStatus: status }),
  setPollingEnabled: (enabled) => set({ pollingEnabled: enabled }),
  setGenerateImage: (value) => set({ generateImage: value }),
  addAttachedFile: (file) => set((state) => ({ attachedFiles: [...state.attachedFiles, file] })),
  removeAttachedFile: (name) => set((state) => ({ attachedFiles: state.attachedFiles.filter((f) => f.name !== name) })),
  clearAttachedFiles: () => set({ attachedFiles: [] }),
  appendStreamingContent: (token) => set((state) => ({ streamingContent: state.streamingContent + token })),
  finalizeStreaming: (content) => set((state) => ({
    messages: [...state.messages, { id: Date.now(), role: 'assistant' as const, content, created_at: new Date().toISOString() }],
    streamingContent: '',
    isStreaming: false,
    isLoading: false,
  })),
  startPolling: () => set({ pollingEnabled: true, executionStatus: 'queued' }),
  stopPolling: () => set({ pollingEnabled: false }),
  reset: () => set({ ...initialState }),
});

export const useGeneralChatStore = createStore(createGeneralChatStore);
