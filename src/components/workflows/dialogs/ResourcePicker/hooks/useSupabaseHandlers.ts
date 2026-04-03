import { useCallback } from 'react';
import type { IntegrationResource } from '@/lib/api-client';
import type { IntegrationType } from '@/lib/integrations/client';
import type { UseToastReturn } from '@/hooks/utility/use-toast';
import { bindOAuthProject } from './supabaseBindingHelpers';
import { useWorkflowSave } from '@/hooks/useWorkflowSave';

interface HandlerConfig {
  supabaseToolNames: string[];
  getConnectionId: (provider: string) => string | null;
  connectIntegration: (provider: IntegrationType, options?: { toolNames: string[]; forceNewAccount?: boolean }) => Promise<string | null>;
  toast: UseToastReturn['toast'];
  setBindingProjectId: (id: string | null) => void;
  setBindModalOpen: (open: boolean) => void;
}

type BindSuccessCallback = (
  resource: IntegrationResource,
  fallback?: { displayName?: string; projectRef?: string }
) => void;

interface BindContext {
  onSuccess: BindSuccessCallback;
  refetchResources: () => void;
  toast: UseToastReturn['toast'];
}

async function executeOAuthBind(
  projectRef: string,
  displayName: string,
  connectionId: string | undefined,
  context: BindContext
) {
  const response = await bindOAuthProject({ projectRef, connectionId });
  context.onSuccess(response.resource, { displayName, projectRef });
  await context.refetchResources();
  context.toast({
    title: 'Project bound',
    description: `${response.resource.name || displayName || projectRef} is ready for Supabase tools.`,
  });
}

export function useSupabaseHandlers(config: HandlerConfig) {
  const { saveWorkflow, hasWorkflow } = useWorkflowSave();

  const handleConnectSupabase = useCallback(async () => {
    try {
      if (hasWorkflow) {
        await saveWorkflow();
      }
      const toolNames = config.supabaseToolNames.length
        ? config.supabaseToolNames
        : ['supabase_table_query'];
      const redirectUrl = await config.connectIntegration('supabase', { toolNames, forceNewAccount: true });
      if (redirectUrl) {
        window.location.href = redirectUrl;
      }
    } catch (error) {
      config.toast({
        title: 'Unable to start Supabase OAuth',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  }, [config, hasWorkflow, saveWorkflow]);

  const handleBindProject = useCallback(
    async (
      projectId: number,
      projectRef: string,
      displayName: string,
      onSuccess: BindSuccessCallback,
      refetchResources: () => void
    ) => {
      config.setBindingProjectId(String(projectId));
      try {
        await executeOAuthBind(projectRef, displayName, config.getConnectionId('supabase') || undefined, {
          onSuccess,
          refetchResources,
          toast: config.toast,
        });
        config.setBindModalOpen(false);
      } catch (error) {
        config.toast({
          title: 'Failed to bind project',
          description: error instanceof Error ? error.message : 'Unable to bind Supabase project.',
          variant: 'destructive',
        });
      } finally {
        config.setBindingProjectId(null);
      }
    },
    [config]
  );

  return {
    handleConnectSupabase,
    handleBindProject,
  };
}
