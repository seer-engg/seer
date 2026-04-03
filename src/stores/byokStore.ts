import { byokApi, type LLMApiKey, type LLMProvider, type TestConnectionResult } from "@/lib/byok-api";
import { createStore } from "./createStore";

export interface BYOKState {
  keys: LLMApiKey[];
  loading: boolean;
  testResult: TestConnectionResult | null;
  fetchKeys: () => Promise<void>;
  addKey: (label: string, apiKey: string, provider?: LLMProvider, baseUrl?: string) => Promise<void>;
  deleteKey: (keyId: number) => Promise<void>;
  testConnection: (apiKey?: string, baseUrl?: string) => Promise<void>;
}

export const useBYOKStore = createStore<BYOKState>((set) => ({
  keys: [],
  loading: false,
  testResult: null,

  fetchKeys: async () => {
    set({ loading: true });
    try {
      const keys = await byokApi.listKeys();
      set({ keys, loading: false });
    } catch {
      set({ loading: false });
    }
  },

  addKey: async (label, apiKey, provider, baseUrl) => {
    await byokApi.addKey(label, apiKey, provider, baseUrl);
    const keys = await byokApi.listKeys();
    set({ keys });
  },

  deleteKey: async (keyId) => {
    await byokApi.deleteKey(keyId);
    const keys = await byokApi.listKeys();
    set({ keys });
  },

  testConnection: async (apiKey, baseUrl) => {
    set({ testResult: null });
    const result = await byokApi.testConnection(apiKey, baseUrl);
    set({ testResult: result });
  },
}));
