import { backendApiClient } from "./api-client";

// ============================================================================
// Knowledge Base Types
// ============================================================================

export type DocumentProcessingStatus = "pending" | "processing" | "completed" | "failed";

export interface KnowledgeBase {
  kb_id: string;
  name: string;
  description?: string;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeDocument {
  doc_id: string;
  name: string;
  mime_type: string;
  file_size: number;
  processing_status: DocumentProcessingStatus;
  processing_error?: string;
  chunk_count: number;
  created_at: string;
}

export interface KnowledgeBaseListResponse {
  items: KnowledgeBase[];
}

export interface DocumentListResponse {
  items: KnowledgeDocument[];
}

export type CreateKnowledgeBasePayload = {
  name: string;
  description?: string;
};

export type UpdateKnowledgeBasePayload = {
  name?: string;
  description?: string;
};

// ============================================================================
// Query Types
// ============================================================================

export interface QueryRequest {
  query: string;
  top_k?: number; // 1-50, default 5
  min_score?: number; // 0.0-1.0, default 0.7
}

export interface QueryResult {
  chunk_id: number;
  doc_id: string;
  doc_name: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  results: QueryResult[];
  query: string;
  kb_id: string;
}

// ============================================================================
// Knowledge Base API
// ============================================================================

export const knowledgeApi = {
  /**
   * List all knowledge bases for the authenticated user
   */
  async listKnowledgeBases(): Promise<KnowledgeBaseListResponse> {
    return backendApiClient.request<KnowledgeBaseListResponse>("/api/v1/knowledge-bases");
  },

  /**
   * Get a specific knowledge base by ID
   */
  async getKnowledgeBase(kbId: string): Promise<KnowledgeBase> {
    return backendApiClient.request<KnowledgeBase>(`/api/v1/knowledge-bases/${kbId}`);
  },

  /**
   * Create a new knowledge base
   */
  async createKnowledgeBase(payload: CreateKnowledgeBasePayload): Promise<KnowledgeBase> {
    return backendApiClient.request<KnowledgeBase>("/api/v1/knowledge-bases", {
      method: "POST",
      body: { name: payload.name, description: payload.description },
    });
  },

  /**
   * Update an existing knowledge base
   */
  async updateKnowledgeBase(kbId: string, payload: UpdateKnowledgeBasePayload): Promise<KnowledgeBase> {
    return backendApiClient.request<KnowledgeBase>(`/api/v1/knowledge-bases/${kbId}`, {
      method: "PUT",
      body: { name: payload.name, description: payload.description },
    });
  },

  /**
   * Delete a knowledge base
   */
  async deleteKnowledgeBase(kbId: string): Promise<void> {
    await backendApiClient.request<void>(`/api/v1/knowledge-bases/${kbId}`, {
      method: "DELETE",
    });
  },

  /**
   * List all documents in a knowledge base
   */
  async listDocuments(kbId: string): Promise<DocumentListResponse> {
    return backendApiClient.request<DocumentListResponse>(`/api/v1/knowledge-bases/${kbId}/documents`);
  },

  /**
   * Upload a document to a knowledge base
   * Uses multipart/form-data for file upload
   */
  async uploadDocument(kbId: string, file: File): Promise<KnowledgeDocument> {
    const formData = new FormData();
    formData.append("file", file);

    return backendApiClient.request<KnowledgeDocument>(`/api/v1/knowledge-bases/${kbId}/documents`, {
      method: "POST",
      body: formData,
    });
  },

  /**
   * Delete a document from a knowledge base
   */
  async deleteDocument(kbId: string, docId: string): Promise<void> {
    await backendApiClient.request<void>(`/api/v1/knowledge-bases/${kbId}/documents/${docId}`, {
      method: "DELETE",
    });
  },

  /**
   * Query a knowledge base for relevant content
   */
  async queryKnowledgeBase(kbId: string, request: QueryRequest): Promise<QueryResponse> {
    return backendApiClient.request<QueryResponse>(`/api/v1/knowledge-bases/${kbId}/query`, {
      method: "POST",
      body: request as unknown as Record<string, unknown>,
    });
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Get accepted file types for document upload
 */
export const ACCEPTED_DOCUMENT_TYPES = ".pdf,.txt,.md,.docx";

/**
 * Check if a file type is accepted
 */
export function isAcceptedFileType(file: File): boolean {
  const acceptedExtensions = [".pdf", ".txt", ".md", ".docx"];
  const fileName = file.name.toLowerCase();
  return acceptedExtensions.some((ext) => fileName.endsWith(ext));
}
