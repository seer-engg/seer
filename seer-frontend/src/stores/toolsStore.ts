import type { JsonObject } from '@/types/workflow-spec';
import type { IntegrationType } from '@/lib/integrations/client';

export interface ToolMetadata {
  name: string;
  description: string;
  required_scopes: string[];
  integration_type?: string | null;
  provider?: string | null;
  output_schema?: JsonObject | null;
  parameters: {
    type: string;
    properties: Record<string, JsonObject>;
    required: string[];
  };
}

export interface ToolIntegrationStatus {
  tool: ToolMetadata;
  integrationType: IntegrationType | null;
  isConnected: boolean;
  connectionId: string | null;
  requiredScopes: string[];
}

export interface IntegrationGroup {
  integration_type: string;
  tools: ToolMetadata[];
  display_name: string;
  description: string;
  isConnected: boolean;
}
