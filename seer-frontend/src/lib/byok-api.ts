import { backendApiClient } from "./api-client";

export type LLMProvider = "openrouter" | "openai" | "anthropic" | "google" | "custom";

export interface LLMApiKey {
  id: number;
  label: string;
  base_url: string;
  provider: LLMProvider;
  key_masked: string;
  is_active: boolean;
  status: string;
  last_used_at: string | null;
  created_at: string;
}

export interface TestConnectionResult {
  success: boolean;
  models: string[];
  error: string | null;
}

export const byokApi = {
  async listKeys(): Promise<LLMApiKey[]> {
    return backendApiClient.request<LLMApiKey[]>("/api/byok/keys");
  },

  async addKey(label: string, apiKey: string, provider: LLMProvider = "openrouter", baseUrl?: string): Promise<LLMApiKey> {
    return backendApiClient.request<LLMApiKey>("/api/byok/keys", {
      method: "POST",
      body: { label, api_key: apiKey, provider, base_url: baseUrl || "https://openrouter.ai/api/v1" },
    });
  },

  async deleteKey(keyId: number): Promise<void> {
    await backendApiClient.request(`/api/byok/keys/${keyId}`, { method: "DELETE" });
  },

  async testConnection(apiKey?: string, baseUrl?: string): Promise<TestConnectionResult> {
    return backendApiClient.request<TestConnectionResult>("/api/byok/keys/test", {
      method: "POST",
      body: { api_key: apiKey || null, base_url: baseUrl || null },
    });
  },
};
