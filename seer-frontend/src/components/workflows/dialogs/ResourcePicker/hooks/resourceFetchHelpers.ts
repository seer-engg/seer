import type { JsonRecord, ResourceBrowseResponse, ResourceItem } from '../types';
import { backendApiClient } from '@/lib/api-client';

export function resolveField(obj: JsonRecord | undefined, path?: string): unknown {
  if (!obj || !path) return undefined;
  return path.split('.').reduce<unknown>((acc, segment) => {
    if (acc === null || acc === undefined) {
      return undefined;
    }
    if (typeof acc !== 'object') {
      return undefined;
    }
    const current = acc as JsonRecord;
    return current[segment as keyof JsonRecord];
  }, obj);
}

function extractItemValue(raw: JsonRecord, valueField: string): unknown {
  return resolveField(raw, valueField) ?? raw?.id ?? raw?.value;
}

function extractItemDisplay(raw: JsonRecord, displayField: string, value: unknown): string {
  return String(
    resolveField(raw, displayField) ??
    raw?.display_name ??
    raw?.name ??
    raw?.title ??
    raw?.resource_key ??
    value
  );
}

function extractItemDescription(raw: JsonRecord): string | undefined {
  const description =
    raw?.description ??
    raw?.metadata?.project_ref ??
    raw?.resource_key ??
    resolveField(raw, 'description');
  return description ? String(description) : undefined;
}

function normalizeItem(raw: JsonRecord, valueField: string, displayField: string): ResourceItem | null {
  const value = extractItemValue(raw, valueField);
  if (value === null || value === undefined) {
    return null;
  }
  const display = extractItemDisplay(raw, displayField, value);
  const description = extractItemDescription(raw);

  return {
    id: String(value),
    name: display,
    display_name: display,
    type: raw?.type || 'resource',
    description,
    raw,
  } as ResourceItem;
}

export function normalizeCustomItems(
  rawItems: JsonRecord[],
  valueField: string,
  displayField: string
): ResourceItem[] {
  return rawItems
    .map((raw) => normalizeItem(raw, valueField, displayField))
    .filter((item): item is ResourceItem => Boolean(item));
}

export function buildQueryParams(config: {
  debouncedQuery: string;
  searchEnabled: boolean;
  normalizedEndpoint: string | null;
  currentParentId?: string;
  pageParam?: string;
  dependsOnParam?: string;
}): URLSearchParams {
  const params = new URLSearchParams();
  const { debouncedQuery, searchEnabled, normalizedEndpoint, currentParentId, pageParam, dependsOnParam } = config;

  if (debouncedQuery && searchEnabled !== false) params.set('q', debouncedQuery);
  if (!normalizedEndpoint && currentParentId) params.set('parent_id', currentParentId);
  if (pageParam) params.set('page_token', pageParam);
  if (dependsOnParam) params.set('depends_on', dependsOnParam);
  if (!normalizedEndpoint) {
    params.set('page_size', '50');
  }

  return params;
}

export function buildEndpointWithParams(baseEndpoint: string, params: URLSearchParams): string {
  const queryString = params.toString();
  return queryString
    ? `${baseEndpoint}${baseEndpoint.includes('?') ? '&' : '?'}${queryString}`
    : baseEndpoint;
}

export async function fetchResourcePage(config: {
  baseEndpoint: string;
  params: URLSearchParams;
}): Promise<ResourceBrowseResponse | JsonRecord | JsonRecord[]> {
  const { baseEndpoint, params } = config;
  const endpointWithParams = buildEndpointWithParams(baseEndpoint, params);
  return backendApiClient.request<ResourceBrowseResponse | JsonRecord | JsonRecord[]>(
    endpointWithParams,
    { method: 'GET' }
  );
}

export function processCustomEndpointResponse(
  response: ResourceBrowseResponse | JsonRecord | JsonRecord[],
  valueField: string,
  displayField: string,
  searchEnabled: boolean
): ResourceBrowseResponse {
  const itemsArray = Array.isArray(response)
    ? response
    : Array.isArray((response as JsonRecord)?.items)
      ? (response as JsonRecord).items as JsonRecord[]
      : [];
  return {
    items: normalizeCustomItems(itemsArray, valueField, displayField),
    next_page_token: (response as JsonRecord)?.next_page_token as string | undefined,
    supports_hierarchy: false,
    supports_search: searchEnabled ?? true,
  };
}
