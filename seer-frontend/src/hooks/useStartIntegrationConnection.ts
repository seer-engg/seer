import { useCallback, useState } from 'react';

import { toast } from '@/components/ui/sonner';
import type { IntegrationType } from '@/lib/integrations/client';

import { useConnectIntegration } from './useConnectIntegration';
import { useWorkflowSave } from './useWorkflowSave';

interface UseStartIntegrationConnectionOptions {
  integrationType: IntegrationType;
  label: string;
  toolNames: string[];
  forceNewAccount?: boolean;
}

export function useStartIntegrationConnection({
  integrationType,
  label,
  toolNames,
  forceNewAccount = true,
}: UseStartIntegrationConnectionOptions) {
  const connectIntegration = useConnectIntegration();
  const { saveWorkflow, hasWorkflow } = useWorkflowSave();
  const [isConnecting, setIsConnecting] = useState(false);

  const startConnection = useCallback(async () => {
    setIsConnecting(true);

    try {
      if (hasWorkflow) {
        await saveWorkflow();
      }

      const redirectUrl = await connectIntegration(integrationType, {
        toolNames,
        forceNewAccount,
      });

      if (redirectUrl) {
        window.location.href = redirectUrl;
        return;
      }

      toast.error(`Unable to start ${label} connection`);
    } catch (error) {
      console.error(`Failed to connect ${label}`, error);
      toast.error(`Unable to start ${label} connection`, {
        description: error instanceof Error ? error.message : 'Please try again.',
      });
    } finally {
      setIsConnecting(false);
    }
  }, [connectIntegration, forceNewAccount, hasWorkflow, integrationType, label, saveWorkflow, toolNames]);

  return {
    startConnection,
    isConnecting,
  };
}
