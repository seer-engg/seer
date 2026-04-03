/* eslint-disable max-lines */
export type TokenProvider = () => Promise<string | null>;
type JsonBody = Record<string, unknown> | unknown[];

interface BackendAPIClientOptions {
  baseUrl: string | (() => string);
  tokenProvider?: TokenProvider;
}

interface BackendRequestInit extends Omit<RequestInit, "body" | "headers"> {
  body?: RequestInit["body"] | JsonBody;
  headers?: HeadersInit;
}

export class BackendAPIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public response?: unknown,
  ) {
    super(message);
    this.name = "BackendAPIError";
  }
}

/**
 * Structured error with node context for workflow validation errors
 */
export interface NodeError {
  code: string;
  message: string;
  node_id: string;
  location?: string;
  expression?: string;
}

/**
 * RFC 7807 Problem Details format used by the backend for validation errors
 */
export interface ProblemDetails {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  errors?: NodeError[] | string[];
}

/**
 * Extract structured error details from a BackendAPIError
 * Returns null if the error doesn't contain RFC 7807 Problem Details
 */
export function extractProblemDetails(error: unknown): ProblemDetails | null {
  if (!(error instanceof BackendAPIError) || !error.response) {
    return null;
  }

  const response = error.response as { detail?: unknown };
  if (typeof response.detail !== "object" || response.detail === null) {
    return null;
  }

  const problemDetails = response.detail as ProblemDetails;
  if (!problemDetails.title && !problemDetails.detail) {
    return null;
  }

  return problemDetails;
}

/**
 * Extract structured node errors from a BackendAPIError
 * Returns empty array if the error doesn't contain node-specific errors
 */
export function extractNodeErrors(error: unknown): NodeError[] {
  const problemDetails = extractProblemDetails(error);
  if (!problemDetails?.errors?.length) return [];

  const first = problemDetails.errors[0];
  if (typeof first === 'object' && first !== null && 'node_id' in first) {
    return problemDetails.errors as NodeError[];
  }
  return [];
}

const ensureAbsoluteUrl = (url: string): string => {
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  if (url.includes("localhost") || url.startsWith("127.")) return `http://${url}`;
  return `https://${url}`;
};

const BACKEND_URL_STORAGE_KEY = "seer-backend-base-url";
const FALLBACK_BACKEND_URL = "http://localhost:8000";

export const getDefaultBackendBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_BACKEND_API_URL;
  return envUrl ? ensureAbsoluteUrl(envUrl) : FALLBACK_BACKEND_URL;
};

export const getStoredBackendBaseUrl = (): string | null => {
  if (typeof window === "undefined") return null;
  const stored = window.localStorage.getItem(BACKEND_URL_STORAGE_KEY);
  if (!stored) return null;

  try {
    const parsed = JSON.parse(stored);
    if (typeof parsed === "string" && parsed.trim()) {
      return ensureAbsoluteUrl(parsed.trim());
    }
  } catch {
    // Ignore parse errors and fall back to raw value
  }

  const trimmed = stored.trim();
  return trimmed ? ensureAbsoluteUrl(trimmed) : null;
};

export const setStoredBackendBaseUrl = (url: string | null): string | null => {
  if (typeof window === "undefined") return url ? ensureAbsoluteUrl(url) : null;
  if (!url || !url.trim()) {
    window.localStorage.removeItem(BACKEND_URL_STORAGE_KEY);
    return null;
  }
  const normalized = ensureAbsoluteUrl(url.trim());
  window.localStorage.setItem(BACKEND_URL_STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
};

/**
 * Gets the backend base URL dynamically.
 * Priority:
 * 1. 'backend' query parameter (for quick self-hosted overrides)
 * 2. Persisted custom URL from settings (localStorage)
 * 3. VITE_BACKEND_API_URL or localhost fallback
 * @returns The backend base URL
 */
export const getBackendBaseUrl = (): string => {
  // Check for backend URL in query parameter (for self-hosted backend)
  if (typeof window !== "undefined") {
    const urlParams = new URLSearchParams(window.location.search);
    const backendParam = urlParams.get("backend");
    if (backendParam) {
      return ensureAbsoluteUrl(backendParam);
    }

    const storedUrl = getStoredBackendBaseUrl();
    if (storedUrl) {
      return storedUrl;
    }
  }
  
  const defaultUrl = getDefaultBackendBaseUrl();

  if (!import.meta.env.VITE_BACKEND_API_URL && defaultUrl === FALLBACK_BACKEND_URL && typeof window !== "undefined") {
    console.warn(
      `[BackendAPIClient] No backend URL configured. Using fallback: ${defaultUrl}. ` +
      `Set VITE_BACKEND_API_URL, configure a custom API URL in Settings, or use ?backend=<url> query parameter.`
    );
  }

  return defaultUrl;
};

const defaultTokenProvider: TokenProvider = async () => {
  if (typeof window === "undefined") return null;
  const clerk = (window as WindowWithClerk).Clerk;
  if (!clerk?.session?.getToken) return null;
  try {
    return await clerk.session.getToken({template:"user-profile"});
  } catch (error) {
    console.warn("Failed to fetch Clerk token", error);
    return null;
  }
};

export const backendTokenProvider = defaultTokenProvider;
const shouldSerializeBody = (value: unknown): value is JsonBody => {
  if (value === null || typeof value !== "object") {
    return false;
  }

  if (value instanceof FormData || value instanceof URLSearchParams || value instanceof Blob) {
    return false;
  }

  if (value instanceof ArrayBuffer || ArrayBuffer.isView(value)) {
    return false;
  }

  if (typeof ReadableStream !== "undefined" && value instanceof ReadableStream) {
    return false;
  }

  return true;
};

export class BackendAPIClient {
  private baseUrl: string | (() => string);
  private tokenProvider?: TokenProvider;

  constructor(options: BackendAPIClientOptions) {
    // Support both static URL and dynamic getter function
    this.baseUrl = typeof options.baseUrl === "function" 
      ? options.baseUrl 
      : ensureAbsoluteUrl(options.baseUrl);
    this.tokenProvider = options.tokenProvider;
  }

  /**
   * Gets the current base URL, resolving it dynamically if it's a function
   */
  private getBaseUrl(): string {
    return typeof this.baseUrl === "function" 
      ? this.baseUrl() 
      : this.baseUrl;
  }

  setTokenProvider(provider: TokenProvider) {
    this.tokenProvider = provider;
  }

  private async getToken(): Promise<string | null> {
    const provider = this.tokenProvider ?? defaultTokenProvider;
    return provider ? provider() : null;
  }

  private toAbsoluteUrl(endpoint: string): string {
    const normalized = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
    return `${this.getBaseUrl()}${normalized}`;
  }

  private parseErrorPayload(responseBody: string): unknown {
    if (!responseBody) return undefined;
    try {
      return JSON.parse(responseBody);
    } catch {
      return responseBody;
    }
  }

  private extractErrorMessage(errorPayload: unknown, response: Response): string {
    if (
      typeof errorPayload === "object" &&
      errorPayload !== null &&
      "detail" in errorPayload &&
      typeof (errorPayload as { detail?: unknown }).detail === "string"
    ) {
      return (errorPayload as { detail: string }).detail;
    }
    return response.statusText ?? `HTTP ${response.status}`;
  }

  private validateContentType(
    contentType: string,
    responseBody: string,
    url: string,
    status: number
  ): void {
    if (contentType.includes("application/json")) return;

    if (responseBody.trim().startsWith("<!doctype") || responseBody.includes("<html")) {
      throw new BackendAPIError(
        `Received HTML instead of JSON from ${url}. Check backend availability.`,
        status,
        { url, contentType }
      );
    }

    throw new BackendAPIError(
      `Unsupported content type: ${contentType || "unknown"}`,
      status,
      { url, contentType, body: responseBody }
    );
  }

  private addCacheControlHeaders(headers: Headers, method?: string): void {
    const isGetRequest = method === 'GET' || !method;
    if (isGetRequest && !headers.has('Cache-Control')) {
      headers.set("Cache-Control", "no-cache, no-store, must-revalidate");
      headers.set("Pragma", "no-cache");
      headers.set("Expires", "0");
    }
  }

  /**
   * Makes a streaming request and returns the raw Response for SSE consumption.
   * Identical to request() but skips JSON parsing — caller reads the body stream.
   */
  async stream(endpoint: string, options: BackendRequestInit = {}): Promise<Response> {
    const url = this.toAbsoluteUrl(endpoint);
    const token = await this.getToken();
    const { body: requestBody, headers: providedHeaders, ...restOptions } = options;

    let body = requestBody as RequestInit["body"] | undefined;
    if (requestBody && shouldSerializeBody(requestBody)) {
      body = JSON.stringify(requestBody);
    }

    const headers = new Headers(providedHeaders || {});
    if (!(body instanceof FormData)) {
      headers.set("Content-Type", headers.get("Content-Type") ?? "application/json");
    }
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const emulatedUserId = localStorage.getItem("dev_emulate_user_id");
    if (import.meta.env.VITE_ENABLE_USER_EMULATION === "true" && emulatedUserId) {
      headers.set("X-Emulate-User-Id", emulatedUserId);
    }
    // SSE streams must not be cached
    headers.set("Cache-Control", "no-cache");
    headers.set("Accept", "text/event-stream");

    const response = await fetch(url, {
      ...restOptions,
      headers,
      body,
    });

    if (!response.ok) {
      const responseBody = await response.text();
      const errorPayload = this.parseErrorPayload(responseBody);
      const message = this.extractErrorMessage(errorPayload, response);
      const error = new BackendAPIError(message, response.status, errorPayload);

      if (response.status === 402) {
        window.dispatchEvent(new CustomEvent('api-error', { detail: error }));
      }

      throw error;
    }

    return response;
  }

  // eslint-disable-next-line complexity
  async request<T>(endpoint: string, options: BackendRequestInit = {}): Promise<T> {
    const url = this.toAbsoluteUrl(endpoint);
    const token = await this.getToken();
    const { body: requestBody, headers: providedHeaders, ...restOptions } = options;

    let body = requestBody as RequestInit["body"] | undefined;
    if (requestBody && shouldSerializeBody(requestBody)) {
      body = JSON.stringify(requestBody);
    }

    const headers = new Headers(providedHeaders || {});
    if (!(body instanceof FormData)) {
      headers.set("Content-Type", headers.get("Content-Type") ?? "application/json");
    }
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const emulatedUserId = localStorage.getItem("dev_emulate_user_id");
    if (import.meta.env.VITE_ENABLE_USER_EMULATION === "true" && emulatedUserId) {
      headers.set("X-Emulate-User-Id", emulatedUserId);
    }
    this.addCacheControlHeaders(headers, restOptions.method);

    const response = await fetch(url, {
      ...restOptions,
      headers,
      body,
      signal: restOptions.signal, // Pass through abort signal
    });

    const contentType = response.headers.get("content-type") || "";
    const responseBody = await response.text();

    if (!response.ok) {
      // Detect HTML responses (common with gateway/proxy errors)
      if (responseBody.trim().startsWith("<!doctype") || responseBody.includes("<html")) {
        throw new BackendAPIError(
          `Received HTML instead of JSON from ${url}. Check backend availability.`,
          response.status,
          { url, contentType }
        );
      }

      const errorPayload = this.parseErrorPayload(responseBody);
      const message = this.extractErrorMessage(errorPayload, response);
      const error = new BackendAPIError(message, response.status, errorPayload);

      // Dispatch custom event for 402 errors to enable global handling
      if (response.status === 402) {
        window.dispatchEvent(new CustomEvent('api-error', { detail: error }));
      }

      throw error;
    }

    if (!responseBody) {
      return undefined as T;
    }

    this.validateContentType(contentType, responseBody, url, response.status);

    try {
      return JSON.parse(responseBody) as T;
    } catch (error) {
      throw new BackendAPIError(
        "Failed to parse JSON response",
        response.status,
        { url, rawBody: responseBody, error: error instanceof Error ? error.message : error }
      );
    }
  }
}

// Create client with dynamic URL resolution - reads URL from query param or env var on each request
export const backendApiClient = new BackendAPIClient({
  baseUrl: getBackendBaseUrl, // Pass function reference for dynamic resolution
});

interface WindowWithClerk extends Window {
  Clerk?: {
    session?: {
      getToken: (options?: Record<string, unknown>) => Promise<string | null>;
    };
  };
}

// ============================================================================
// Integration & Tool Execution Types and Functions
// ============================================================================

export interface ConnectedAccount {
  id: string;
  status: "ACTIVE" | "PENDING" | "INACTIVE";
  user_id?: string;
  toolkit: {
    slug: string;
  };
  scopes?: string;
  provider?: string;
}

export interface ListConnectedAccountsResponse {
  items: ConnectedAccount[];
  total: number;
}

/**
 * Tool connection status - indicates whether a specific tool has the required OAuth scopes
 */
export interface ToolConnectionStatus {
  tool_name: string;
  integration_type: string | null;
  provider: string | null;
  connected: boolean;
  has_required_scopes: boolean;
  access_token_valid?: boolean;  // Whether access token exists and is not expired
  has_refresh_token?: boolean;  // Whether refresh_token exists (needed for token refresh)
  missing_scopes: string[];
  connection_id: string | null;
  provider_account_id?: string;
  requires_oauth_connection?: boolean;
  requires_secrets?: boolean;
  supports_tokenless_auth?: boolean;
  auth_mode?: "none" | "oauth" | "secrets" | "oauth_and_secrets";
}

export interface ToolsConnectionStatusResponse {
  tools: ToolConnectionStatus[];
}

// ============================================================================
// Multi-Account OAuth Types (for account selector in tool config)
// ============================================================================

/**
 * Tool-specific account info for account selector dropdown.
 * Shows which accounts are available for a specific tool with scope validation.
 */
export interface ToolAccountInfo {
  id: number;
  provider_account_id: string;
  display_name: string;  // User-friendly name (e.g., email address)
  has_required_scopes: boolean;
  missing_scopes: string[];
}

/**
 * Response from GET /api/integrations/tools/{tool_name}/accounts
 * Used to populate the account selector dropdown in tool config panels.
 */
export interface ToolAccountsResponse {
  tool_name: string;
  provider: string | null;
  accounts: ToolAccountInfo[];
  requires_selection: boolean;  // true when 2+ accounts exist and user must choose
  required_scopes: string[];  // OAuth scopes required by this tool (for initiating new connections)
}

/**
 * Account info for settings page (grouped by provider).
 */
export interface AccountInfo {
  id: number;
  provider: string;
  provider_account_id: string;
  display_name?: string | null;
  status: string;
  scopes: string | null;
}

/**
 * Response from GET /api/integrations/connections
 * Returns all connections grouped by provider for the settings page.
 */
export interface ConnectionsByProvider {
  connections: Record<string, AccountInfo[]>;
}

/**
 * Integration status - shows connection status for an integration type
 */

export interface ConnectResponse {
  redirectUrl: string;
  connectionId: string;
}

export interface WaitForConnectionResponse {
  status: string;
  connectionId: string;
}

export interface ExecuteToolResponse {
  data: unknown;
  success: boolean;
  error?: string;
}

export interface IntegrationResource {
  id: number;
  provider: string;
  resource_type: string;
  resource_id: string;
  resource_key?: string | null;
  name?: string | null;
  status: string;
  metadata?: Record<string, unknown>;
  oauth_connection_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface IntegrationSecret {
  id: number;
  provider: string;
  name: string;
  secret_type: string;
  resource_id?: number | null;
  oauth_connection_id?: string | null;
  value_fingerprint?: string | null;
  metadata?: Record<string, unknown>;
  status: string;
  expires_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface SupabaseBindingResponse {
  resource: IntegrationResource;
  secrets: IntegrationSecret[];
}

/**
 * List connected accounts for a user
 */
export async function listConnectedAccounts(params: {
  userIds?: string[];
  toolkitSlugs?: string[];
  authConfigIds?: string[];
}): Promise<ListConnectedAccountsResponse> {
  // Backend uses authenticated user from JWT token, so user_id query parameter is not needed
  const endpoint = `/api/integrations/`;
  const response = await backendApiClient.request<{ items: ConnectedAccount[] }>(endpoint);

  // Filter by toolkitSlugs if provided
  let items = response.items || [];
  if (params.toolkitSlugs && params.toolkitSlugs.length > 0) {
    items = items.filter((item) => params.toolkitSlugs!.includes(item.toolkit.slug));
  }

  return {
    items,
    total: items.length,
  };
}

/**
 * Get connection status for all tools.
 * This is the primary way to check which tools are connected and have required scopes.
 */
export async function getToolsConnectionStatus(): Promise<ToolsConnectionStatusResponse> {
  const endpoint = `/api/integrations/tools/status`;
  return backendApiClient.request<ToolsConnectionStatusResponse>(endpoint);
}

/**
 * Single tool status response for connection-specific queries.
 * Used by canvas tool blocks to check status for their configured connection.
 */
export interface SingleToolStatusResponse {
  tool_name: string;
  integration_type: string | null;
  provider: string | null;
  supports_oauth: boolean;
  supports_manual_secrets: boolean;
  connected: boolean;
  missing_scopes: string[];
  connection_id: string | null;
  provider_account_id: string | null;
}

/**
 * Get connection status for a specific tool, optionally for a specific connection.
 *
 * Use this when you need to check a tool's status for a particular OAuth connection
 * (e.g., when a workflow tool block has a specific connection_id configured).
 *
 * @param toolName - The tool name (e.g., 'gmail_send_email')
 * @param connectionId - Optional specific OAuth connection ID to validate against
 * @returns Tool status for the specified tool/connection combination
 */
export async function getToolStatusForConnection(
  toolName: string,
  connectionId?: number | null,
): Promise<SingleToolStatusResponse> {
  const encoded = encodeURIComponent(toolName);
  const params = connectionId != null ? `?connection_id=${connectionId}` : '';
  return backendApiClient.request<SingleToolStatusResponse>(
    `/api/integrations/tools/${encoded}/status${params}`,
  );
}

/**
 * Initiate OAuth connection
 * CRITICAL: Frontend must always pass scope parameter
 *
 * Builds OAuth redirect URL with JWT token for authentication.
 * OAuth flows require browser navigation (not fetch) because the backend
 * redirects to the OAuth provider, which doesn't allow cross-origin requests.
 * 
 * The caller should navigate using: window.location.href = response.redirectUrl
 */
export async function initiateConnection(params: {
  userId: string;
  provider: string;
  scope: string; // REQUIRED - OAuth scopes (frontend controls)
  callbackUrl?: string;
  integrationType?: string; // Integration type that triggered this (e.g., 'gmail', 'google_sheets')
  forceNewAccount?: boolean; // When true, skips existing scope check to allow adding another account
  connectionId?: number; // Specific OAuth connection ID for scope checks (multi-account support)
}): Promise<ConnectResponse> {
  const searchParams = new URLSearchParams();
  searchParams.append("user_id", params.userId);
  searchParams.append("scope", params.scope);

  if (params.callbackUrl) {
    searchParams.append("redirect_to", params.callbackUrl);
  }

  // Pass integration type so backend can track which tool triggered this connection
  if (params.integrationType) {
    searchParams.append("integration_type", params.integrationType);
  }

  // Force new account flow - skips existing scope check to allow adding another OAuth account
  if (params.forceNewAccount) {
    searchParams.append("force_new_account", "true");
  }

  // Pass specific connection ID for accurate scope checks (multi-account support)
  if (params.connectionId != null) {
    searchParams.append("connection_id", String(params.connectionId));
  }

  // Include JWT token for backend authentication
  // OAuth flows require browser navigation, so we can't use Authorization header
  const token = await backendTokenProvider();
  if (token) {
    searchParams.append("token", token);
  }

  // Build full backend URL for OAuth redirect
  // The browser will navigate directly to this URL
  const backendUrl = getBackendBaseUrl();
  const redirectUrl = `${backendUrl}/api/integrations/${params.provider}/connect?${searchParams.toString()}`;

  return {
    redirectUrl,
    connectionId: "pending", // Will be set after OAuth callback
  };
}

/**
 * Wait for OAuth connection to be established
 */
export async function waitForConnection(params: {
  connectionId: string;
  timeoutMs?: number;
}): Promise<WaitForConnectionResponse> {
  // Polling logic - TODO: implement actual polling
  return { status: "ACTIVE", connectionId: params.connectionId };
}

/**
 * Delete a connected account
 */
export async function deleteConnectedAccount(accountId: string): Promise<void> {
  await backendApiClient.request<void>(`/api/integrations/${accountId}`, {
    method: "DELETE",
  });
}

/**
 * Get all connections grouped by provider.
 * Used by the settings page to display connected accounts organized by provider.
 */
export async function getConnectionsByProvider(): Promise<ConnectionsByProvider> {
  return backendApiClient.request<ConnectionsByProvider>('/api/integrations/connections');
}

/**
 * Get available accounts for a specific tool.
 * Used by the account selector dropdown in tool config panels.
 * Returns accounts with scope validation status.
 *
 * @param toolName - The tool name (e.g., 'gmail_send_email')
 * @returns Tool accounts info including whether selection is required
 */
export async function getToolAccounts(toolName: string): Promise<ToolAccountsResponse> {
  const encoded = encodeURIComponent(toolName);
  return backendApiClient.request<ToolAccountsResponse>(`/api/integrations/tools/${encoded}/accounts`);
}

/**
 * Get available OAuth accounts for a specific trigger.
 * Used by frontend to let users select which account to use for OAuth-based triggers.
 * Response structure is identical to ToolAccountsResponse.
 * @param triggerKey - The trigger key (e.g., 'poll.gmail.email_received')
 * @returns Trigger accounts info including whether selection is required
 */
export async function getTriggerAccounts(triggerKey: string): Promise<ToolAccountsResponse> {
  const encoded = encodeURIComponent(triggerKey);
  return backendApiClient.request<ToolAccountsResponse>(`/api/v1/triggers/${encoded}/accounts`);
}

/**
 * Execute a tool
 */
export async function executeTool(params: {
  toolSlug: string;
  userId: string;
  connectionId?: string;
  arguments?: Record<string, unknown>;
}): Promise<ExecuteToolResponse> {
  const endpoint = `/api/tools/${params.toolSlug}/execute`;

  try {
    const response = await backendApiClient.request<ExecuteToolResponse>(endpoint, {
      method: "POST",
      body: {
        user_id: params.userId,
        connection_id: params.connectionId,
        arguments: params.arguments || {},
      },
    });

    return response;
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Tool execution failed";
    return {
      success: false,
      data: null,
      error: errorMessage,
    };
  }
}

/**
 * List persisted Supabase project bindings for the authenticated user.
 */
export async function listSupabaseBindings(params?: {
  resourceType?: string;
}): Promise<IntegrationResource[]> {
  const searchParams = new URLSearchParams();
  if (params?.resourceType) {
    searchParams.set('resource_type', params.resourceType);
  }
  const query = searchParams.toString();
  const endpoint = `/api/integrations/supabase/resources/bindings${query ? `?${query}` : ''}`;
  const response = await backendApiClient.request<{ items: IntegrationResource[] }>(endpoint);
  return response.items || [];
}

/**
 * Bind a Supabase project by project reference and optional connection ID.
 */
export async function bindSupabaseProject(params: {
  projectRef: string;
  connectionId?: string;
}): Promise<SupabaseBindingResponse> {
  return backendApiClient.request<SupabaseBindingResponse>('/api/integrations/supabase/projects/bind', {
    method: 'POST',
    body: {
      project_ref: params.projectRef,
      connection_id: params.connectionId,
    },
  });
}

/**
 * Bind a Supabase project by directly supplying secrets instead of OAuth.
 * Mirrors SupabaseManualBindRequest on the backend.
 */
export async function bindSupabaseProjectManual(params: {
  projectRef: string;
  projectName?: string;
  serviceRoleKey?: string;
  anonKey?: string;
  connectionId?: string;
}): Promise<SupabaseBindingResponse> {
  return backendApiClient.request<SupabaseBindingResponse>(
    '/api/integrations/supabase/projects/manual-bind',
    {
      method: 'POST',
      body: {
        project_ref: params.projectRef,
        project_name: params.projectName,
        service_role_key: params.serviceRoleKey,
        anon_key: params.anonKey,
        connection_id: params.connectionId,
      },
    },
  );
}

/**
 * Delete (revoke) a persisted integration resource binding.
 */
export async function deleteIntegrationResource(resourceId: number): Promise<void> {
  await backendApiClient.request(`/api/integrations/resources/${resourceId}`, {
    method: 'DELETE',
  });
}

// ============================================================================
// Postgres Database Binding Types and Functions
// ============================================================================

export interface PostgresTestRequest {
  connection_string?: string;
  host?: string;
  port?: number;
  database?: string;
  username?: string;
  password?: string;
  ssl_mode?: 'disable' | 'allow' | 'prefer' | 'require' | 'verify-ca' | 'verify-full';
}

export interface PostgresTestResponse {
  status: string;
  connected: boolean;
  database?: string;
  user?: string;
  server_version?: string;
  error?: string;
}

export interface PostgresBindRequest extends PostgresTestRequest {
  name: string;
  access_mode?: 'restricted' | 'unrestricted';
}

export interface PostgresBindingResponse {
  resource: IntegrationResource;
  secrets: IntegrationSecret[];
}

/**
 * Test a Postgres database connection without saving it.
 */
export async function testPostgresConnection(
  params: PostgresTestRequest,
): Promise<PostgresTestResponse> {
  return backendApiClient.request<PostgresTestResponse>(
    '/api/integrations/postgres/databases/test',
    {
      method: 'POST',
      body: params as unknown as Record<string, unknown>,
    },
  );
}

/**
 * Bind a Postgres database by connection details.
 */
export async function bindPostgresDatabase(
  params: PostgresBindRequest,
): Promise<PostgresBindingResponse> {
  return backendApiClient.request<PostgresBindingResponse>(
    '/api/integrations/postgres/databases/bind',
    {
      method: 'POST',
      body: params as unknown as Record<string, unknown>,
    },
  );
}

/**
 * Update an existing Postgres database binding.
 * If password is omitted, existing credentials are preserved.
 */
export async function updatePostgresDatabase(
  resourceId: number,
  params: PostgresBindRequest,
): Promise<PostgresBindingResponse> {
  return backendApiClient.request<PostgresBindingResponse>(
    `/api/integrations/postgres/databases/${resourceId}`,
    {
      method: 'PATCH',
      body: params as unknown as Record<string, unknown>,
    },
  );
}

/**
 * List persisted Postgres database bindings for the authenticated user.
 */
export async function listPostgresBindings(): Promise<IntegrationResource[]> {
  const endpoint = '/api/integrations/postgres/resources/bindings';
  const response = await backendApiClient.request<{ items: IntegrationResource[] }>(endpoint);
  return response.items || [];
}

/**
 * Check if a resource exists and get its status.
 */
export async function getResourceStatus(resourceId: number | string): Promise<{
  exists: boolean;
  status: string | null;
}> {
  return await backendApiClient.request<{ exists: boolean; status: string | null }>(
    `/api/integrations/resources/${resourceId}/status`,
  );
}

// ============================================================================
// Discovery Chat Functions
// ============================================================================

import type { ClarificationAnswers } from '@/types/discovery';

/**
 * Resume workflow chat after a clarification question interrupt.
 * Returns raw Response for SSE stream consumption.
 */
export async function resumeWorkflowChat(params: {
  workflowId: string;
  threadId: string;
  answers?: ClarificationAnswers;
  message?: string;
  requestId: string;
}): Promise<Response> {
  const endpoint = `/api/nexus/${params.workflowId}/chat/resume`;

  return backendApiClient.stream(endpoint, {
    method: 'POST',
    body: {
      thread_id: params.threadId,
      answers: params.answers,
      message: params.message,
      request_id: params.requestId,
    },
  });
}


/**
 * User settings types
 */
export interface UserSettings {
  max_agent_steps: number | null;
  timezone: string | null;
  preferences: Record<string, unknown>;
  updated_at: string;
}

/**
 * Get current user's settings
 */
export async function getUserSettings(): Promise<UserSettings> {
  return backendApiClient.request<UserSettings>('/api/users/me/settings');
}

/**
 * Update current user's settings
 */
export async function updateUserSettings(update: {
  per_run_cost_cap_usd?: number;
  max_agent_steps?: number;
  timezone?: string;
  preferences?: Record<string, unknown>;
}): Promise<UserSettings> {
  return backendApiClient.request<UserSettings>('/api/users/me/settings', {
    method: 'PATCH',
    body: update,
  });
}

// ============================================================================
// Webhook Start Listening & Pending Events
// ============================================================================

export interface StartListeningResponse {
  webhook_url?: string;
  secret_token?: string;
  subscription_id: number;
  form_url?: string;
}

export interface PendingEventItem {
  event_id: number;
  data: Record<string, unknown>;
  received_at: string;
}

export interface PendingEventsResponse {
  events: PendingEventItem[];
  latest_event_id: number | null;
}

/**
 * Provision a webhook subscription immediately (without publish).
 * Returns the webhook URL and signing secret the user can use to trigger events.
 */
export async function startListening(workflowId: string, triggerId: string): Promise<StartListeningResponse> {
  return backendApiClient.request<StartListeningResponse>(
    `/api/v1/workflows/${workflowId}/triggers/${triggerId}/start-listening`,
    { method: 'POST' },
  );
}

/**
 * Poll for webhook events received since a given event ID.
 * Used by the frontend while in "Start Listening" mode.
 */
export async function getPendingEvents(
  workflowId: string,
  triggerId: string,
  since?: number,
): Promise<PendingEventsResponse> {
  const params = since != null ? `?since=${since}` : '';
  return backendApiClient.request<PendingEventsResponse>(
    `/api/v1/workflows/${workflowId}/triggers/${triggerId}/pending-events${params}`,
  );
}

// ============================================================================
// Trigger Event Browsing (Real Events for Testing)
// ============================================================================

import type {
  TriggerEventsResponse,
  TriggerSubscriptionListItemsResponse,
  TriggerSubscriptionFilters,
} from '@/types/triggers';

export interface FetchTriggerEventsParams {
  /** Provider name (e.g., 'google', 'discord', 'generic', 'form') */
  provider: string;
  /** Trigger key (e.g., 'poll.gmail.email_received', 'webhook.generic') */
  triggerKey: string;
  /** Provider connection ID for polling triggers (gmail, discord) */
  providerConnectionId?: number;
  /** Subscription ID for persisted triggers (webhooks, forms) */
  subscriptionId?: number;
  /** Optional pagination token */
  pageToken?: string;
  /** Optional filter params JSON */
  filterParams?: Record<string, unknown>;
}

/**
 * Fetch real trigger events from a user's connected account or stored events.
 * Used for testing workflows with actual data (emails, messages, webhooks, forms)
 *
 * For polling triggers: provide providerConnectionId
 * For persisted triggers: provide subscriptionId
 */
export async function fetchTriggerEvents(
  params: FetchTriggerEventsParams,
): Promise<TriggerEventsResponse> {
  const searchParams = new URLSearchParams();

  // Add the appropriate ID based on trigger type
  if (params.providerConnectionId !== undefined) {
    searchParams.set('provider_connection_id', String(params.providerConnectionId));
  }
  if (params.subscriptionId !== undefined) {
    searchParams.set('subscription_id', String(params.subscriptionId));
  }
  if (params.pageToken) {
    searchParams.set('page_token', params.pageToken);
  }
  if (params.filterParams) {
    searchParams.set('filter_params', JSON.stringify(params.filterParams));
  }

  return backendApiClient.request<TriggerEventsResponse>(
    `/api/integrations/resources/${params.provider}/trigger_events/${params.triggerKey}?${searchParams}`,
  );
}

export interface SubscriptionEventCountResponse {
  subscription_id: number;
  event_count: number;
  has_events: boolean;
}

/**
 * Get the count of stored events for a trigger subscription.
 * Used by the frontend to determine if "Browse events" should be shown
 * for persisted triggers (webhooks, forms).
 */
export async function getSubscriptionEventCount(
  subscriptionId: number,
): Promise<SubscriptionEventCountResponse> {
  return backendApiClient.request<SubscriptionEventCountResponse>(
    `/api/v1/subscriptions/${subscriptionId}/event-count`,
  );
}

// ============================================================================
// Trigger Subscription Management
// ============================================================================

/**
 * List trigger subscriptions with extended info for management view.
 * Includes workflow title and last event timestamp.
 */
export async function listTriggerSubscriptions(
  filters?: TriggerSubscriptionFilters,
): Promise<TriggerSubscriptionListItemsResponse> {
  const searchParams = new URLSearchParams();
  if (filters?.workflow_id) {
    searchParams.set('workflow_id', filters.workflow_id);
  }
  if (filters?.trigger_key) {
    searchParams.set('trigger_key', filters.trigger_key);
  }
  if (filters?.search) {
    searchParams.set('search', filters.search);
  }
  const query = searchParams.toString();
  return backendApiClient.request<TriggerSubscriptionListItemsResponse>(
    `/api/v1/trigger-subscriptions${query ? `?${query}` : ''}`,
  );
}

/**
 * Toggle the enabled status of a trigger subscription.
 */
export async function updateTriggerSubscriptionEnabled(
  subscriptionId: number,
  enabled: boolean,
): Promise<void> {
  await backendApiClient.request(
    `/api/v1/trigger-subscriptions/${subscriptionId}`,
    {
      method: 'PATCH',
      body: { enabled },
    },
  );
}

// ============================================================================
// Trigger Event Generation (Instant Triggering)
// ============================================================================

export interface TriggerEventGenerateResponse {
  envelope: Record<string, unknown>;
  display_title: string;
}

/**
 * Generate a synthetic trigger event for immediate workflow execution.
 * Used for triggers like cron that don't have historical events to browse.
 *
 * @param triggerKey - The trigger key (e.g., 'schedule.cron')
 * @param triggerId - The trigger instance ID from the workflow
 * @param providerConfig - Trigger configuration (e.g., cron_expression, timezone)
 * @returns Generated event envelope ready for workflow execution
 */
// ============================================================================
// Global Variables
// ============================================================================

export interface GlobalVariableItem {
  id: number;
  key: string;
  value: string;
  is_secret: boolean;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface GlobalVariableListResponse {
  items: GlobalVariableItem[];
}

export async function listGlobalVariables(): Promise<GlobalVariableListResponse> {
  return backendApiClient.request<GlobalVariableListResponse>('/api/v1/variables');
}

export async function createGlobalVariable(data: {
  key: string;
  value: string;
  is_secret?: boolean;
  description?: string;
}): Promise<GlobalVariableItem> {
  return backendApiClient.request<GlobalVariableItem>('/api/v1/variables', {
    method: 'POST',
    body: data,
  });
}

export async function updateGlobalVariable(
  variableId: number,
  data: { value?: string; is_secret?: boolean; description?: string },
): Promise<GlobalVariableItem> {
  return backendApiClient.request<GlobalVariableItem>(`/api/v1/variables/${variableId}`, {
    method: 'PATCH',
    body: data,
  });
}

export async function deleteGlobalVariable(variableId: number): Promise<void> {
  await backendApiClient.request(`/api/v1/variables/${variableId}`, {
    method: 'DELETE',
  });
}

export async function generateTriggerEvent(
  triggerKey: string,
  triggerId: string,
  providerConfig: Record<string, unknown> = {},
): Promise<TriggerEventGenerateResponse> {
  const encodedKey = encodeURIComponent(triggerKey);
  return backendApiClient.request<TriggerEventGenerateResponse>(
    `/api/v1/triggers/${encodedKey}/generate-event`,
    {
      method: 'POST',
      body: {
        trigger_id: triggerId,
        provider_config: providerConfig,
      },
    },
  );
}

// ============================================================================
// Nexus Chat Streaming Functions
// ============================================================================

/**
 * Stream nexus chat response via SSE
 */
export async function streamNexusChat(
  params: {
    workflowId: string;
    message: string;
    model: string;
    sessionId?: number;
    threadId?: string;
    workflowState?: unknown;
  },
  onToken: (token: string) => void,
  onComplete: (data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const token = await backendTokenProvider();
  const baseUrl = getBackendBaseUrl();
  const response = await fetch(`${baseUrl}/api/nexus/${params.workflowId}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      message: params.message,
      model: params.model,
      session_id: params.sessionId,
      thread_id: params.threadId,
      workflow_state: params.workflowState,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.error) throw new Error(data.error);
        if (data.done) {
          onComplete(data);
          return;
        }
        if (data.token) onToken(data.token);
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }
}

/**
 * Stream nexus chat resume response via SSE
 */
export async function streamNexusResume(
  params: {
    workflowId: string;
    threadId: string;
    answers: ClarificationAnswers;
  },
  onToken: (token: string) => void,
  onComplete: (data: Record<string, unknown>) => void,
  signal?: AbortSignal,
): Promise<void> {
  const token = await backendTokenProvider();
  const baseUrl = getBackendBaseUrl();
  const response = await fetch(`${baseUrl}/api/nexus/${params.workflowId}/chat/resume/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({
      thread_id: params.threadId,
      answers: params.answers,
    }),
    signal,
  });

  if (!response.ok) {
    throw new Error(`Stream resume request failed: ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const data = JSON.parse(line.slice(6));
        if (data.error) throw new Error(data.error);
        if (data.done) {
          onComplete(data);
          return;
        }
        if (data.token) onToken(data.token);
      } catch (e) {
        if (e instanceof SyntaxError) continue;
        throw e;
      }
    }
  }
}
