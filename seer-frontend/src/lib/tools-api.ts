import { backendApiClient, getToolsConnectionStatus, listConnectedAccounts } from '@/lib/api-client';
import type { FunctionBlockSchema } from '@/components/workflows/types';
import type {
  ConnectedAccount,
  ToolConnectionStatus,
} from '@/lib/api-client';

export interface ToolCatalogItem {
  name: string;
  description: string;
  required_scopes: string[];
  integration_type?: string | null;
  provider?: string | null;
  output_schema?: Record<string, unknown> | null;
  parameters: {
    type: string;
    properties: Record<string, Record<string, unknown>>;
    required: string[];
  };
}

interface NodeFieldDescriptor {
  name: string;
  kind: string;
  required?: boolean;
}

interface NodeTypeDescriptor {
  type: string;
  title: string;
  fields: NodeFieldDescriptor[];
}

interface NodeTypeResponse {
  node_types: NodeTypeDescriptor[];
}

export async function listToolCatalog() {
  const response = await backendApiClient.request<{ tools: ToolCatalogItem[] }>('/api/tools', {
    method: 'GET',
  });

  return response.tools;
}

export async function listToolConnectionStatus(): Promise<ToolConnectionStatus[]> {
  const response = await getToolsConnectionStatus();
  return response.tools || [];
}

export async function listConnections(): Promise<ConnectedAccount[]> {
  const response = await listConnectedAccounts({});
  return response.items;
}

export async function listFunctionBlocks(): Promise<FunctionBlockSchema[]> {
  const response = await backendApiClient.request<NodeTypeResponse>(
    '/api/v1/builder/node-types',
    { method: 'GET' },
  );

  return response.node_types.map<FunctionBlockSchema>((nodeType) => ({
    type: nodeType.type as FunctionBlockSchema['type'],
    label: nodeType.title,
    category: 'Core',
    description: `${nodeType.title} block`,
    defaults: {},
    config_schema: {
      type: 'object',
      properties: nodeType.fields.reduce<Record<string, unknown>>((acc, field) => {
        acc[field.name] = { type: 'string', description: field.kind };
        return acc;
      }, {}),
    },
  }));
}
