/**
 * Integration Metadata Store
 *
 * Centralized store for dynamic integration metadata fetched from the backend.
 * This enables the frontend to be fully dynamic - adding a new integration
 * in the backend is sufficient without frontend code changes.
 *
 * The store provides:
 * - Display names and icons for integrations
 * - OAuth provider mappings
 * - Scope display names and descriptions
 * - Detection patterns for tool name → integration type mapping
 */
import type { StateCreator } from 'zustand';

import { backendApiClient } from '@/lib/api-client';

import { createStore } from './createStore';

// ============================================================================
// Types
// ============================================================================

export interface IntegrationIcon {
  type: 'url' | 'lucide' | 'svg';
  value: string;
}

export interface IntegrationScope {
  value: string;
  display_name: string;
  description?: string;
}

export interface IntegrationDetectionPatterns {
  tool_name_patterns?: string[];
  scope_keywords?: string[];
}

export interface IntegrationMetadata {
  type: string;
  display_name: string;
  oauth_provider: string | null;
  requires_oauth: boolean;
  icon: IntegrationIcon | null;
  brand_color?: string;
  default_scopes: string[];
  scopes: IntegrationScope[];
  detection_patterns?: IntegrationDetectionPatterns;
}

export interface IntegrationMetadataResponse {
  integrations: IntegrationMetadata[];
  provider_to_types: Record<string, string[]>;
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Detect integration type from metadata patterns
 */
function detectFromMetadataPatterns(
  metadata: Map<string, IntegrationMetadata>,
  lowerToolName: string,
  scopes: string[],
): string | null {
  for (const integration of metadata.values()) {
    const patterns = integration.detection_patterns;
    if (!patterns) continue;

    // Check tool name patterns
    const matchedByName = patterns.tool_name_patterns?.some((pattern) =>
      lowerToolName.includes(pattern.toLowerCase()),
    );
    if (matchedByName) return integration.type;

    // Check scope keywords
    const matchedByScope = patterns.scope_keywords?.some((keyword) =>
      scopes.some((scope) => scope.toLowerCase().includes(keyword.toLowerCase())),
    );
    if (matchedByScope) return integration.type;
  }
  return null;
}

/**
 * Format scope display name
 */
function formatScopeDisplayName(scope: string): string {
  // Google scopes - extract service name
  if (scope.includes('googleapis.com')) {
    const match = scope.match(/\/auth\/([^/]+)$/);
    if (match) return match[1];
  }

  // Capitalize simple scopes like 'email', 'profile'
  if (scope === 'email' || scope === 'profile' || scope === 'openid') {
    return scope.charAt(0).toUpperCase() + scope.slice(1);
  }

  return scope;
}

// ============================================================================
// Store Interface
// ============================================================================

export interface IntegrationMetadataStore {
  // State
  metadata: Map<string, IntegrationMetadata>;
  providerToTypes: Map<string, string[]>;
  loading: boolean;
  loaded: boolean;
  error: string | null;

  // Actions
  loadMetadata: () => Promise<void>;

  // Selectors
  getMetadata: (type: string) => IntegrationMetadata | null;
  getOAuthProvider: (type: string) => string | null;
  getDisplayName: (type: string) => string;
  getIcon: (type: string) => IntegrationIcon | null;
  getScopeDisplayName: (scope: string) => string;
  getProviderDisplayName: (provider: string) => string;
  requiresOAuth: (type: string) => boolean;
  getDefaultScopes: (type: string) => string[];
  detectIntegrationType: (toolName: string, scopes: string[]) => string | null;
  getAllIntegrationTypes: () => string[];
}

// ============================================================================
// Store Implementation
// ============================================================================

const createIntegrationMetadataStore: StateCreator<IntegrationMetadataStore> = (set, get) => ({
  // Initial state
  metadata: new Map(),
  providerToTypes: new Map(),
  loading: false,
  loaded: false,
  error: null,

  // Actions
  async loadMetadata() {
    const state = get();
    if (state.loading || state.loaded) return;

    set({ loading: true, error: null });

    try {
      const response = await backendApiClient.request<IntegrationMetadataResponse>(
        '/api/integrations/metadata',
        { method: 'GET' },
      );

      const metadataMap = new Map<string, IntegrationMetadata>();
      for (const integration of response.integrations) {
        metadataMap.set(integration.type, integration);
      }

      const providerToTypesMap = new Map<string, string[]>();
      for (const [provider, types] of Object.entries(response.provider_to_types)) {
        providerToTypesMap.set(provider, types);
      }

      set({ metadata: metadataMap, providerToTypes: providerToTypesMap, loading: false, loaded: true });
    } catch (error) {
      console.error('[IntegrationMetadataStore] Failed to load metadata:', error);
      set({
        loading: false,
        loaded: false,
        error: error instanceof Error ? error.message : 'Failed to load integration metadata',
      });
    }
  },

  // Selectors
  getMetadata(type: string) {
    return get().metadata.get(type) ?? null;
  },

  getOAuthProvider(type: string) {
    return get().metadata.get(type)?.oauth_provider ?? null;
  },

  getDisplayName(type: string) {
    return get().metadata.get(type)?.display_name ?? type;
  },

  getIcon(type: string) {
    return get().metadata.get(type)?.icon ?? null;
  },

  getScopeDisplayName(scope: string) {
    // Check if any loaded integration has this scope with a display name
    const { metadata } = get();
    for (const integration of metadata.values()) {
      const scopeInfo = integration.scopes.find((s) => s.value === scope);
      if (scopeInfo) return scopeInfo.display_name;
    }
    // Format the scope as a fallback
    return formatScopeDisplayName(scope);
  },

  getProviderDisplayName(provider: string) {
    // Find an integration that uses this provider and derive display name
    const { metadata } = get();
    for (const integration of metadata.values()) {
      if (integration.oauth_provider === provider) {
        // For compound names like "Gmail", extract provider name
        // Or just capitalize the provider
        return provider.charAt(0).toUpperCase() + provider.slice(1);
      }
    }
    return provider.charAt(0).toUpperCase() + provider.slice(1);
  },

  requiresOAuth(type: string) {
    return get().metadata.get(type)?.requires_oauth ?? true;
  },

  getDefaultScopes(type: string) {
    return get().metadata.get(type)?.default_scopes ?? [];
  },

  detectIntegrationType(toolName: string, scopes: string[]) {
    const { metadata } = get();
    const lowerToolName = toolName.toLowerCase();
    return detectFromMetadataPatterns(metadata, lowerToolName, scopes);
  },

  getAllIntegrationTypes() {
    return Array.from(get().metadata.keys());
  },
});

export const useIntegrationMetadataStore = createStore(createIntegrationMetadataStore);
