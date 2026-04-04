import { Check, AlertTriangle } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { DynamicIntegrationIcon } from '@/components/ui/DynamicIntegrationIcon';
import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';
import type { RequiredIntegration, TemplateRequirementsResponse } from '@/types/templates';

export interface IntegrationRequirementsProps {
  requirements: TemplateRequirementsResponse;
}

interface IntegrationItemProps {
  integration: RequiredIntegration;
  isConnected: boolean;
}

function IntegrationItem({ integration, isConnected }: IntegrationItemProps) {
  return (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center gap-3">
        <DynamicIntegrationIcon
          integrationType={integration.integration_type}
          width={20}
          height={20}
        />
        <div>
          <p className="text-sm font-medium">
            {useIntegrationMetadataStore.getState().getDisplayName(integration.integration_type)}
          </p>
          <p className="text-xs text-muted-foreground">{integration.reason}</p>
        </div>
      </div>
      {isConnected ? (
        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/20">
          <Check className="w-3 h-3 mr-1" />
          Connected
        </Badge>
      ) : (
        <Badge variant="outline" className="bg-amber-500/10 text-amber-600 border-amber-500/20">
          <AlertTriangle className="w-3 h-3 mr-1" />
          Not Connected
        </Badge>
      )}
    </div>
  );
}

/**
 * Displays the integration requirements for a template.
 * Shows which integrations are connected vs missing.
 */
export function IntegrationRequirements({ requirements }: IntegrationRequirementsProps) {
  const connected = requirements?.connected ?? [];
  const missing = requirements?.missing ?? [];
  const hasAnyRequirements = connected.length > 0 || missing.length > 0;

  if (!hasAnyRequirements) {
    return (
      <div className="text-sm text-muted-foreground">
        This template has no integration requirements.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {missing.length > 0 && (
        <Alert variant="default" className="border-amber-500/30 bg-amber-500/5">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-sm">
            {missing.length === 1
              ? 'You need to connect 1 integration before using this template.'
              : `You need to connect ${missing.length} integrations before using this template.`}
            {' '}You can still create the workflow and connect them later.
          </AlertDescription>
        </Alert>
      )}

      <div className="divide-y">
        {connected.map((integration) => (
          <IntegrationItem
            key={`${integration.provider}-${integration.integration_type}`}
            integration={integration}
            isConnected={true}
          />
        ))}
        {missing.map((integration) => (
          <IntegrationItem
            key={`${integration.provider}-${integration.integration_type}`}
            integration={integration}
            isConnected={false}
          />
        ))}
      </div>
    </div>
  );
}
