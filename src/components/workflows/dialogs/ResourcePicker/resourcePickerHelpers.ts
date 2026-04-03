import type { IntegrationResource } from '@/lib/api-client';
import type { ResourceItem } from './types';

export function normalizeEndpoint(endpoint?: string): string | null {
  if (!endpoint) return null;
  if (endpoint.startsWith('http')) return endpoint;
  if (endpoint.startsWith('/api')) return endpoint;
  const sanitized = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `/api${sanitized}`;
}

export function isSupabaseBindingEndpoint(endpoint: string | null): boolean {
  if (!endpoint) return false;
  return endpoint.includes('/api/integrations/supabase/resources/bindings');
}

export function isPostgresBindingEndpoint(endpoint: string | null): boolean {
  if (!endpoint) return false;
  return endpoint.includes('/api/integrations/postgres/resources/bindings');
}

export function buildBaseEndpoint(
  normalizedEndpoint: string | null,
  provider?: string,
  resourceType?: string
): string | null {
  if (normalizedEndpoint) return normalizedEndpoint;
  if (!provider) return null;
  return `/api/integrations/resources/${provider}/${resourceType}`;
}

export function buildResourceItem(
  resource: IntegrationResource,
  fallback?: { displayName?: string; projectRef?: string }
): ResourceItem {
  const metadataProjectRef =
    typeof resource.metadata?.project_ref === 'string'
      ? (resource.metadata.project_ref as string)
      : undefined;
  const display =
    resource.name ||
    resource.resource_key ||
    metadataProjectRef ||
    fallback?.displayName ||
    fallback?.projectRef ||
    resource.id;
  const description = metadataProjectRef || resource.resource_key || fallback?.projectRef;

  return {
    id: String(resource.id),
    name: String(display),
    display_name: String(display),
    type: 'binding',
    description: description ? String(description) : undefined,
    resource_key: resource.resource_key || undefined,
  };
}
