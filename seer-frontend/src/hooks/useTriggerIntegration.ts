import { useCallback, useMemo } from 'react';
import {
  CALENDAR_TRIGGER_KEY,
  CALENDAR_START_TRIGGER_KEY,
  DISCORD_TRIGGER_KEY,
  GMAIL_TRIGGER_KEY,
  GOOGLE_SHEETS_TRIGGER_KEY,
  SLACK_TRIGGER_KEY,
} from '@/components/workflows/triggers/constants';
import { useTriggersStore } from '@/stores/triggersStore';
import type { WorkflowNodeData } from '@/components/workflows/types';
import type { IntegrationType } from '@/lib/integrations/client';

import { useActiveWorkflowId } from './useActiveWorkflowId';
import { useCalendarIntegration } from './useCalendarIntegration';
import { useDiscordIntegration } from './useDiscordIntegration';
import { useGmailIntegration } from './useGmailIntegration';
import { useSlackIntegration } from './useSlackIntegration';
import { useStartIntegrationConnection } from './useStartIntegrationConnection';
import { useTriggerCatalogQuery } from './useTriggerCatalogQuery';

type TriggerIntegrationMeta = NonNullable<WorkflowNodeData['triggerMeta']>['integration'];
type TriggerIntegrationContext =
  | TriggerIntegrationMeta['gmail']
  | TriggerIntegrationMeta['discord']
  | TriggerIntegrationMeta['slack']
  | TriggerIntegrationMeta['calendar']
  | TriggerIntegrationMeta['google_sheets']
  | undefined;

export function useTriggerIntegration(triggerKey: string, triggerId: string): TriggerIntegrationContext {
  const workflowId = useActiveWorkflowId();
  const { data: triggerCatalog = [] } = useTriggerCatalogQuery();
  const { gmailToolNames, gmailConnectionId } = useGmailIntegration();
  const { discordToolNames, discordConnectionId } = useDiscordIntegration();
  const { slackToolNames, slackConnectionId } = useSlackIntegration();
  const { calendarToolNames, calendarConnectionId } = useCalendarIntegration();

  // Read connection ID from trigger's stored provider_config
  const storedConnectionId = useTriggersStore((state) => {
    if (!workflowId) return null;
    const triggers = state.workflowTriggers.get(workflowId) ?? [];
    const trigger = triggers.find((t) => t.id === triggerId);
    return (trigger?.provider_config?.provider_connection_id as number | null) ?? null;
  });

  const descriptor = useMemo(
    () => triggerCatalog.find((entry) => entry.key === triggerKey) ?? null,
    [triggerCatalog, triggerKey],
  );

  const handleSelectAccount = useCallback(
    (connectionId: number | null) => {
      if (!workflowId || !triggerKey) {
        return;
      }

      const triggers = useTriggersStore.getState().workflowTriggers.get(workflowId) ?? [];
      const trigger = triggers.find((entry) => entry.id === triggerId);
      const currentProviderConfig = trigger?.provider_config ?? {};

      useTriggersStore.getState().updateTrigger(workflowId, triggerId, {
        provider_config: {
          ...currentProviderConfig,
          provider_connection_id: connectionId,
        },
      });
    },
    [triggerId, triggerKey, workflowId],
  );

  const integrationConfig = useMemo<{
    integrationType: IntegrationType;
    label: string;
    toolNames: string[];
    connectionId: number | null;
  } | null>(() => {
    if (triggerKey === GMAIL_TRIGGER_KEY) {
      return {
        integrationType: 'gmail',
        label: 'Gmail',
        toolNames: gmailToolNames,
        connectionId: gmailConnectionId,
      };
    }

    if (triggerKey === DISCORD_TRIGGER_KEY) {
      return {
        integrationType: 'discord',
        label: 'Discord',
        toolNames: discordToolNames,
        connectionId: discordConnectionId,
      };
    }

    if (triggerKey === SLACK_TRIGGER_KEY) {
      return {
        integrationType: 'slack',
        label: 'Slack',
        toolNames: slackToolNames,
        connectionId: slackConnectionId,
      };
    }

    if (triggerKey === CALENDAR_TRIGGER_KEY || triggerKey === CALENDAR_START_TRIGGER_KEY) {
      return {
        integrationType: 'google_calendar',
        label: 'Google Calendar',
        toolNames: calendarToolNames,
        connectionId: calendarConnectionId,
      };
    }

    if (triggerKey === GOOGLE_SHEETS_TRIGGER_KEY) {
      return {
        integrationType: 'google_sheets' as IntegrationType,
        label: 'Google Sheets',
        toolNames: ['google_sheets_read', 'google_sheets_batch_read'],
        connectionId: null,
      };
    }

    return null;
  }, [
    calendarConnectionId,
    calendarToolNames,
    discordConnectionId,
    discordToolNames,
    gmailConnectionId,
    gmailToolNames,
    slackConnectionId,
    slackToolNames,
    triggerKey,
  ]);

  const { startConnection: beginConnectFlow, isConnecting } = useStartIntegrationConnection({
    integrationType: integrationConfig?.integrationType ?? 'gmail',
    label: integrationConfig?.label ?? 'integration',
    toolNames: integrationConfig?.toolNames ?? [],
  });

  const isReady = descriptor?.is_connected ?? false;

  return useMemo(() => {
    if (triggerKey === GMAIL_TRIGGER_KEY && integrationConfig?.integrationType === 'gmail') {
      return {
        ready: isReady,
        connectionId: storedConnectionId,
        onConnect: beginConnectFlow,
        isConnecting,
        triggerKey,
        onSelectAccount: handleSelectAccount,
      } satisfies NonNullable<TriggerIntegrationMeta['gmail']>;
    }

    if (triggerKey === DISCORD_TRIGGER_KEY && integrationConfig?.integrationType === 'discord') {
      return {
        ready: isReady,
        connectionId: storedConnectionId,
        onConnect: beginConnectFlow,
        isConnecting,
        triggerKey,
        onSelectAccount: handleSelectAccount,
      } satisfies NonNullable<TriggerIntegrationMeta['discord']>;
    }

    if (triggerKey === SLACK_TRIGGER_KEY && integrationConfig?.integrationType === 'slack') {
      return {
        ready: isReady,
        connectionId: storedConnectionId,
        onConnect: beginConnectFlow,
        isConnecting,
        triggerKey,
        onSelectAccount: handleSelectAccount,
      } satisfies NonNullable<TriggerIntegrationMeta['slack']>;
    }

    if ((triggerKey === CALENDAR_TRIGGER_KEY || triggerKey === CALENDAR_START_TRIGGER_KEY) && integrationConfig?.integrationType === 'google_calendar') {
      return {
        ready: isReady,
        connectionId: storedConnectionId,
        onConnect: beginConnectFlow,
        isConnecting,
        triggerKey,
        onSelectAccount: handleSelectAccount,
      } satisfies NonNullable<TriggerIntegrationMeta['calendar']>;
    }

    if (triggerKey === GOOGLE_SHEETS_TRIGGER_KEY) {
      return {
        ready: isReady,
        connectionId: storedConnectionId,
        onConnect: beginConnectFlow,
        isConnecting,
        triggerKey,
        onSelectAccount: handleSelectAccount,
      } satisfies NonNullable<TriggerIntegrationMeta['google_sheets']>;
    }

    return undefined;
  }, [
    beginConnectFlow,
    handleSelectAccount,
    integrationConfig,
    isConnecting,
    isReady,
    storedConnectionId,
    triggerKey,
  ]);
}
