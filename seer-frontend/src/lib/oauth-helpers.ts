/**
 * OAuth helper functions
 *
 * These utilities format OAuth scopes and provider names for display.
 * They now delegate to the integration metadata store for dynamic configuration.
 */

import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';

/**
 * Helper function to format OAuth scope names for display.
 * Delegates to the metadata store for scope display names.
 *
 * @param scope - The raw OAuth scope string (e.g., 'https://www.googleapis.com/auth/gmail.readonly')
 * @returns A human-readable display name (e.g., 'Gmail (read-only)')
 */
export function formatScopeName(scope: string): string {
  return useIntegrationMetadataStore.getState().getScopeDisplayName(scope);
}

/**
 * Helper function to get provider display name.
 * Delegates to the metadata store for provider display names.
 *
 * @param provider - The OAuth provider identifier (e.g., 'google', 'github')
 * @returns A human-readable display name (e.g., 'Google', 'GitHub')
 */
export function getProviderDisplayName(provider: string): string {
  return useIntegrationMetadataStore.getState().getProviderDisplayName(provider);
}
