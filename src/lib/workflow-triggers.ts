import { backendApiClient } from '@/lib/api-client';
import type { TriggerCatalogResponse, TriggerDescriptor } from '@/types/triggers';

export async function listTriggers(): Promise<TriggerDescriptor[]> {
  const response = await backendApiClient.request<TriggerCatalogResponse>('/api/v1/triggers');
  return response.triggers;
}
