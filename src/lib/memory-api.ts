import { backendApiClient } from "./api-client";

// ============================================================================
// Memory Types
// ============================================================================

export interface Memory {
  id: string;
  memory: string;
  score: number | null;
  created_at: string;
  updated_at?: string | null;
  metadata?: {
    source?: string;
    added_at?: string;
    edited_at?: string;
    thread_id?: string;
    session_id?: number;
    workflow_id?: number;
    extracted_at?: string;
    message_count?: number;
  };
}

export interface MemoryListResponse {
  memories: Memory[];
  total: number;
  user_id?: string;
}

export interface MemorySearchResponse {
  memories: Memory[];
  total: number;
  query?: string;
  user_id?: string;
}

export interface MemoryBank {
  memory_bank_id: string;
  name: string;
  description?: string | null;
  is_default: boolean;
  memory_count: number;
  created_at: string;
  updated_at: string;
}

export interface MemoryBankListResponse {
  items: MemoryBank[];
  total: number;
}

export interface MemoryStats {
  total_memories: number;
  memory_enabled: boolean;
  extraction_enabled: boolean;
  injection_enabled: boolean;
}

export interface MemoryMutationResponse {
  memory: Memory;
  message: string;
}

export interface MemoryBankMutationRequest {
  name: string;
  description?: string | null;
}

export interface MemoryDeleteResponse {
  deleted: boolean;
  memory_id?: string | null;
  message: string;
}

// ============================================================================
// Memory API
// ============================================================================

export const memoryApi = {
  async listBanks(): Promise<MemoryBankListResponse> {
    return backendApiClient.request<MemoryBankListResponse>("/api/memory/banks");
  },

  async createBank(payload: MemoryBankMutationRequest): Promise<MemoryBank> {
    return backendApiClient.request<MemoryBank>("/api/memory/banks", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },

  async updateBank(
    memoryBankId: string,
    payload: Partial<MemoryBankMutationRequest>
  ): Promise<MemoryBank> {
    return backendApiClient.request<MemoryBank>(`/api/memory/banks/${memoryBankId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  },

  async deleteBank(memoryBankId: string): Promise<MemoryDeleteResponse> {
    return backendApiClient.request<MemoryDeleteResponse>(`/api/memory/banks/${memoryBankId}`, {
      method: "DELETE",
    });
  },

  async setDefaultBank(memoryBankId: string): Promise<MemoryBank> {
    return backendApiClient.request<MemoryBank>(`/api/memory/banks/${memoryBankId}/set-default`, {
      method: "POST",
    });
  },

  /**
   * List all memories for the authenticated user
   */
  async listMemories(memoryBankId?: string): Promise<MemoryListResponse> {
    if (memoryBankId) {
      return backendApiClient.request<MemoryListResponse>(
        `/api/memory/banks/${memoryBankId}/items`
      );
    }
    return backendApiClient.request<MemoryListResponse>("/api/memory");
  },

  /**
   * Search memories semantically
   */
  async searchMemories(query: string, memoryBankId?: string): Promise<MemorySearchResponse> {
    const params = new URLSearchParams({ q: query });
    if (memoryBankId) {
      return backendApiClient.request<MemorySearchResponse>(
        `/api/memory/banks/${memoryBankId}/items/search?${params}`
      );
    }
    return backendApiClient.request<MemorySearchResponse>(
      `/api/memory/search?${params}`
    );
  },

  /**
   * Get memory statistics and feature status
   */
  async getStats(): Promise<MemoryStats> {
    return backendApiClient.request<MemoryStats>("/api/memory/stats");
  },

  /**
   * Create a new manual memory
   */
  async createMemory(memory: string, memoryBankId?: string): Promise<MemoryMutationResponse> {
    if (memoryBankId) {
      return backendApiClient.request<MemoryMutationResponse>(
        `/api/memory/banks/${memoryBankId}/items`,
        {
          method: "POST",
          body: JSON.stringify({ memory }),
        }
      );
    }
    return backendApiClient.request<MemoryMutationResponse>("/api/memory", {
      method: "POST",
      body: JSON.stringify({ memory }),
    });
  },

  /**
   * Update an existing memory by ID
   */
  async updateMemory(
    memoryId: string,
    memory: string,
    memoryBankId?: string
  ): Promise<MemoryMutationResponse> {
    if (memoryBankId) {
      return backendApiClient.request<MemoryMutationResponse>(
        `/api/memory/banks/${memoryBankId}/items/${memoryId}`,
        {
          method: "PUT",
          body: JSON.stringify({ memory }),
        }
      );
    }
    return backendApiClient.request<MemoryMutationResponse>(`/api/memory/${memoryId}`, {
      method: "PUT",
      body: JSON.stringify({ memory }),
    });
  },

  /**
   * Delete a specific memory by ID
   */
  async deleteMemory(memoryId: string, memoryBankId?: string): Promise<void> {
    if (memoryBankId) {
      await backendApiClient.request<void>(`/api/memory/banks/${memoryBankId}/items/${memoryId}`, {
        method: "DELETE",
      });
      return;
    }
    await backendApiClient.request<void>(`/api/memory/${memoryId}`, {
      method: "DELETE",
    });
  },

  /**
   * Delete ALL memories (GDPR compliance)
   */
  async deleteAllMemories(memoryBankId?: string): Promise<void> {
    if (memoryBankId) {
      const response = await this.listMemories(memoryBankId);
      await Promise.all(
        response.memories.map((memory) => this.deleteMemory(memory.id, memoryBankId))
      );
      return;
    }
    await backendApiClient.request<void>("/api/memory", {
      method: "DELETE",
    });
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format a date string to a relative time display
 */
export function formatMemoryDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;

  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

/**
 * Truncate content to a maximum length with ellipsis
 */
export function truncateContent(content: string, maxLength: number = 150): string {
  if (content.length <= maxLength) return content;
  return content.substring(0, maxLength).trim() + "...";
}
