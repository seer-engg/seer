import { backendApiClient } from "./api-client";

// ============================================================================
// User Files Types
// ============================================================================

export interface UserFile {
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  size_human: string;
  run_id: string | null;
  workflow_id: string | null;
  workflow_name: string | null;
  source_node_id: string | null;
  source_tool: string | null;
  created_at: string;
}

export interface MimeTypeStats {
  mime_type: string;
  file_count: number;
  total_size_bytes: number;
  total_size_human: string;
}

export interface ToolStats {
  source_tool: string;
  file_count: number;
  total_size_bytes: number;
}

export interface StorageStats {
  total_files: number;
  total_size_bytes: number;
  total_size_human: string;
  files_by_mime_type: MimeTypeStats[];
  files_by_tool: ToolStats[];
  oldest_file_date: string | null;
  newest_file_date: string | null;
}

export interface FileListParams {
  limit?: number;
  cursor?: string;
  mime_type?: string;
  filename?: string;
  source_tool?: string;
  created_after?: string;
  created_before?: string;
  min_size_bytes?: number;
  max_size_bytes?: number;
  sort_by?: "created_at" | "size_bytes" | "filename";
  sort_order?: "asc" | "desc";
}

export interface FileListResponse {
  files: UserFile[];
  total_count: number;
  total_size_bytes: number;
  next_cursor: string | null;
}

export interface FileDownloadResponse {
  file_id: string;
  filename: string;
  download_url: string;
  expires_in_seconds: number;
}

export interface BulkDeleteResult {
  file_id: string;
  deleted: boolean;
  error: string | null;
}

export interface BulkDeleteResponse {
  results: BulkDeleteResult[];
  deleted_count: number;
  failed_count: number;
  total_size_freed_bytes: number;
}

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  size_human: string;
  created_at: string;
}

export interface FileSearchResponse {
  files: UserFile[];
  total_count: number;
}

// ============================================================================
// Files API
// ============================================================================

export const filesApi = {
  /**
   * List all files for the authenticated user with optional filtering
   */
  async listFiles(params: FileListParams = {}): Promise<FileListResponse> {
    const searchParams = new URLSearchParams();

    if (params.limit) searchParams.set("limit", String(params.limit));
    if (params.cursor) searchParams.set("cursor", params.cursor);
    if (params.mime_type) searchParams.set("mime_type", params.mime_type);
    if (params.filename) searchParams.set("filename", params.filename);
    if (params.source_tool) searchParams.set("source_tool", params.source_tool);
    if (params.created_after) searchParams.set("created_after", params.created_after);
    if (params.created_before) searchParams.set("created_before", params.created_before);
    if (params.min_size_bytes) searchParams.set("min_size_bytes", String(params.min_size_bytes));
    if (params.max_size_bytes) searchParams.set("max_size_bytes", String(params.max_size_bytes));
    if (params.sort_by) searchParams.set("sort_by", params.sort_by);
    if (params.sort_order) searchParams.set("sort_order", params.sort_order);

    const query = searchParams.toString();
    const endpoint = `/api/v1/files${query ? `?${query}` : ""}`;

    return backendApiClient.request<FileListResponse>(endpoint);
  },

  /**
   * Get storage statistics for the authenticated user
   */
  async getStorageStats(): Promise<StorageStats> {
    return backendApiClient.request<StorageStats>("/api/v1/files/stats");
  },

  /**
   * Get a single file by ID
   */
  async getFile(fileId: string): Promise<UserFile> {
    const response = await backendApiClient.request<{ file: UserFile }>(`/api/v1/files/${fileId}`);
    return response.file;
  },

  /**
   * Get a presigned download URL for a file
   * @param fileId - File UUID
   * @param inline - If true, returns URL for inline preview instead of download
   */
  async getDownloadUrl(fileId: string, inline: boolean = false): Promise<FileDownloadResponse> {
    const params = inline ? "?inline=true" : "";
    return backendApiClient.request<FileDownloadResponse>(`/api/v1/files/${fileId}/download${params}`);
  },

  /**
   * Get file content URL for preview (proxied through backend to avoid CORS)
   * @param fileId - File UUID
   * @returns URL that can be fetched to get file content
   */
  getContentUrl(fileId: string): string {
    return `/api/v1/files/${fileId}/content`;
  },

  /**
   * Delete a single file
   */
  async deleteFile(fileId: string): Promise<void> {
    await backendApiClient.request<void>(`/api/v1/files/${fileId}`, {
      method: "DELETE",
    });
  },

  /**
   * Bulk delete multiple files
   */
  async bulkDeleteFiles(fileIds: string[]): Promise<BulkDeleteResponse> {
    return backendApiClient.request<BulkDeleteResponse>("/api/v1/files/bulk-delete", {
      method: "POST",
      body: { file_ids: fileIds },
    });
  },

  /**
   * Search files by query string
   */
  async searchFiles(query: string, limit?: number): Promise<FileSearchResponse> {
    const searchParams = new URLSearchParams();
    searchParams.set("q", query);
    if (limit) searchParams.set("limit", String(limit));

    return backendApiClient.request<FileSearchResponse>(
      `/api/v1/files/search?${searchParams.toString()}`
    );
  },

  /**
   * Upload a new file
   * Uses multipart/form-data for file upload
   */
  async uploadFile(file: File, filename?: string): Promise<FileUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (filename) {
      formData.append("filename", filename);
    }

    return backendApiClient.request<FileUploadResponse>("/api/v1/files/upload", {
      method: "POST",
      body: formData,
    });
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get file type category from MIME type for preview routing
 */
export type PreviewCategory = "pdf" | "image" | "text" | "data" | "other";

export function getPreviewCategory(mimeType: string): PreviewCategory {
  if (mimeType === "application/pdf") return "pdf";
  if (mimeType.startsWith("image/")) return "image";
  if (
    mimeType === "text/plain" ||
    mimeType === "text/markdown" ||
    mimeType === "text/csv"
  )
    return "text";
  if (
    mimeType === "application/json" ||
    mimeType === "application/xml" ||
    mimeType === "text/xml"
  )
    return "data";
  return "other";
}

/**
 * Check if a file type can be previewed in the browser
 */
export function isPreviewable(mimeType: string): boolean {
  const category = getPreviewCategory(mimeType);
  return category !== "other";
}

/**
 * Get file type category from MIME type
 */
export function getFileTypeCategory(
  mimeType: string
): "pdf" | "image" | "document" | "data" | "other" {
  if (mimeType === "application/pdf") return "pdf";
  if (mimeType.startsWith("image/")) return "image";
  if (
    mimeType.includes("document") ||
    mimeType.includes("word") ||
    mimeType === "text/plain" ||
    mimeType === "text/markdown"
  )
    return "document";
  if (
    mimeType === "application/json" ||
    mimeType.includes("csv") ||
    mimeType.includes("spreadsheet") ||
    mimeType.includes("excel")
  )
    return "data";
  return "other";
}

/**
 * Format bytes to human-readable size
 */
export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/**
 * Get file extension from filename
 */
export function getFileExtension(filename: string): string {
  const lastDot = filename.lastIndexOf(".");
  return lastDot > 0 ? filename.slice(lastDot + 1).toLowerCase() : "";
}

/**
 * Format relative date for display
 */
export function formatRelativeDate(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;

  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}
