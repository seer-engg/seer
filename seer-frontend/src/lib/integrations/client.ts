/**
 * Integration client for OAuth connections
 * Frontend controls OAuth scopes (read-only is core differentiation)
 *
 * NOTE: IntegrationType and OAuthProvider are now dynamic strings.
 * The metadata store (integrationMetadataStore) provides the source of truth
 * for display names, icons, provider mappings, and scope information.
 *
 * Legacy type aliases are kept for backwards compatibility during migration.
 */

import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';

/**
 * Integration type identifier (e.g., 'gmail', 'github', 'supabase')
 * This is now a string type - new integrations can be added in the backend
 * without requiring frontend changes.
 */
export type IntegrationType = string;

/**
 * OAuth provider identifier (e.g., 'google', 'github', 'discord')
 * Multiple integration types can map to the same provider.
 */
export type OAuthProvider = string;

/**
 * Map integration type to OAuth provider.
 * This function now delegates to the integration metadata store,
 * which fetches provider mappings from the backend.
 *
 * @param integrationType - The integration type (e.g., 'gmail', 'google_drive', 'google_sheets')
 * @returns The OAuth provider to use for the connection (e.g., 'google')
 */
export function getOAuthProvider(integrationType: IntegrationType): OAuthProvider | null {
  return useIntegrationMetadataStore.getState().getOAuthProvider(integrationType);
}

/**
 * Format scopes as space-separated string for OAuth request.
 */
export function formatScopes(scopes: string[]): string {
  return scopes.join(" ");
}

