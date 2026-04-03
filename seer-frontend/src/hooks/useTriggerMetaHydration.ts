import { useEffect } from 'react';
import type { Node } from '@xyflow/react';

import type { WorkflowNodeData } from '@/components/workflows/types';
import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import type { TriggerDescriptor } from '@/types/triggers';
import { GMAIL_TRIGGER_KEY, DISCORD_TRIGGER_KEY, SLACK_TRIGGER_KEY, CALENDAR_TRIGGER_KEY } from '@/components/workflows/triggers/constants';

interface IntegrationContext {
  gmailConnectionId: number | null;
  handleConnectGmail: () => void;
  isConnectingGmail: boolean;
  discordConnectionId: number | null;
  handleConnectDiscord: () => void;
  isConnectingDiscord: boolean;
  slackConnectionId: number | null;
  handleConnectSlack: () => void;
  isConnectingSlack: boolean;
  calendarConnectionId: number | null;
  handleConnectCalendar: () => void;
  isConnectingCalendar: boolean;
  // Account selection handler for multi-account OAuth support
  onSelectAccount?: (triggerId: string, connectionId: number | null) => void;
}

type TriggerMeta = NonNullable<WorkflowNodeData['triggerMeta']>;
type IntegrationMeta = NonNullable<TriggerMeta['integration']>;
type IntegrationType = 'gmail' | 'discord' | 'slack' | 'calendar';

// Map trigger keys to integration types
const INTEGRATION_TRIGGER_KEYS: Record<string, IntegrationType> = {
  [GMAIL_TRIGGER_KEY]: 'gmail',
  [DISCORD_TRIGGER_KEY]: 'discord',
  [SLACK_TRIGGER_KEY]: 'slack',
  [CALENDAR_TRIGGER_KEY]: 'calendar',
};

/**
 * Gets trigger key from node - handles both newly created nodes and loaded nodes.
 */
function getTriggerKey(node: Node<WorkflowNodeData>): string | undefined {
  return (
    (node.data.config?.triggerKey as string | undefined) ||
    node.data.triggerMeta?.subscription?.key
  );
}

/**
 * Checks if an integration needs hydration based on type.
 */
function checkIntegrationNeedsHydration(
  integration: IntegrationMeta | undefined,
  integrationType: IntegrationType,
  expectedReady: boolean,
): boolean {
  if (!integration) return true;
  const data = integration[integrationType];
  return !data || data.ready !== expectedReady;
}

/**
 * Checks if a single trigger node needs hydration.
 */
function checkNodeNeedsHydration(
  node: Node<WorkflowNodeData>,
  triggerCatalog: TriggerDescriptor[],
): boolean {
  if (node.type !== 'trigger' || !node.data.triggerMeta) {
    return false;
  }

  const triggerKey = getTriggerKey(node);
  if (!triggerKey) {
    return false;
  }

  const catalogDescriptor = triggerCatalog.find((t) => t.key === triggerKey);
  if (!catalogDescriptor) {
    return false;
  }

  // Check if descriptor is missing
  if (!node.data.triggerMeta.descriptor) {
    return true;
  }

  // Check if integration needs hydration
  const integrationType = INTEGRATION_TRIGGER_KEYS[triggerKey];
  if (integrationType) {
    const expectedReady = catalogDescriptor.is_connected ?? false;
    return checkIntegrationNeedsHydration(node.data.triggerMeta.integration, integrationType, expectedReady);
  }

  return false;
}

/**
 * Creates integration data for a specific type.
 */
function createIntegrationData(
  integrationType: IntegrationType,
  triggerId: string,
  triggerKey: string,
  isConnected: boolean,
  context: IntegrationContext,
) {
  const makeSelectHandler = () => context.onSelectAccount
    ? (connectionId: number | null) => context.onSelectAccount!(triggerId, connectionId)
    : undefined;

  const configs: Record<IntegrationType, () => IntegrationMeta[IntegrationType]> = {
    gmail: () => ({
      ready: isConnected,
      connectionId: context.gmailConnectionId,
      onConnect: context.handleConnectGmail,
      isConnecting: context.isConnectingGmail,
      triggerKey,
      onSelectAccount: makeSelectHandler(),
    }),
    discord: () => ({
      ready: isConnected,
      connectionId: context.discordConnectionId,
      onConnect: context.handleConnectDiscord,
      isConnecting: context.isConnectingDiscord,
      triggerKey,
      onSelectAccount: makeSelectHandler(),
    }),
    slack: () => ({
      ready: isConnected,
      connectionId: context.slackConnectionId,
      onConnect: context.handleConnectSlack,
      isConnecting: context.isConnectingSlack,
      triggerKey,
      onSelectAccount: makeSelectHandler(),
    }),
    calendar: () => ({
      ready: isConnected,
      connectionId: context.calendarConnectionId,
      onConnect: context.handleConnectCalendar,
      isConnecting: context.isConnectingCalendar,
      triggerKey,
      onSelectAccount: makeSelectHandler(),
    }),
  };

  return configs[integrationType]();
}

/**
 * Hydrates a single trigger node with metadata from the catalog.
 * Returns the original node if no changes needed, or a new node with hydrated metadata.
 */
function hydrateNode(
  node: Node<WorkflowNodeData>,
  triggerCatalog: TriggerDescriptor[],
  integrationContext: IntegrationContext,
): Node<WorkflowNodeData> {
  if (node.type !== 'trigger' || !node.data.triggerMeta) {
    return node;
  }

  const triggerKey = getTriggerKey(node);
  if (!triggerKey) {
    return node;
  }

  const catalogDescriptor = triggerCatalog.find((t) => t.key === triggerKey);
  if (!catalogDescriptor) {
    return node;
  }

  let nodeChanged = false;
  let newTriggerMeta = { ...node.data.triggerMeta };

  // Hydrate descriptor if missing
  if (!newTriggerMeta.descriptor) {
    nodeChanged = true;
    newTriggerMeta = { ...newTriggerMeta, descriptor: catalogDescriptor };
  }

  // Hydrate integration data if needed
  const integrationType = INTEGRATION_TRIGGER_KEYS[triggerKey];
  if (integrationType) {
    const isConnected = catalogDescriptor.is_connected ?? false;
    if (checkIntegrationNeedsHydration(newTriggerMeta.integration, integrationType, isConnected)) {
      nodeChanged = true;
      const integrationData = createIntegrationData(integrationType, node.id, triggerKey, isConnected, integrationContext);
      const newIntegration = {
        ...(newTriggerMeta.integration ?? {}),
        [integrationType]: integrationData,
      };
      newTriggerMeta = { ...newTriggerMeta, integration: newIntegration };
    }
  }

  if (!nodeChanged) {
    return node;
  }

  return {
    ...node,
    data: {
      ...node.data,
      triggerMeta: newTriggerMeta,
    },
  };
}

/**
 * Hydrates trigger nodes with complete metadata from the trigger catalog.
 *
 * This hook solves two related bugs:
 *
 * 1. **Missing integration data** - After page refresh, trigger nodes show "Action required"
 *    even when OAuth credentials are connected. This happens because buildGraphFromSpec()
 *    creates trigger nodes without integration data.
 *
 * 2. **Missing descriptor** - After page refresh, trigger configuration forms don't render
 *    because buildGraphFromSpec() only sets `subscription` in triggerMeta, missing the
 *    `descriptor` field which contains config_schema, event_schema, and other metadata
 *    needed to render the configuration UI.
 *
 * @param nodes - Canvas nodes from React Flow
 * @param setNodes - Function to update canvas nodes
 * @param integrationContext - OAuth flow callbacks and connection IDs
 */
export function useTriggerMetaHydration(
  nodes: Node<WorkflowNodeData>[],
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void,
  integrationContext: IntegrationContext,
) {
  const { data: triggerCatalog = [] } = useTriggerCatalogQuery();

  useEffect(() => {
    if (!nodes.length || !triggerCatalog.length) {
      return;
    }

    // Check if any trigger nodes need hydration
    const needsHydration = nodes.some((node) => checkNodeNeedsHydration(node, triggerCatalog));
    if (!needsHydration) {
      return;
    }

    // Hydrate nodes and track changes
    let anyChanges = false;
    const updatedNodes = nodes.map((node) => {
      const hydratedNode = hydrateNode(node, triggerCatalog, integrationContext);
      if (hydratedNode !== node) {
        anyChanges = true;
      }
      return hydratedNode;
    });

    if (anyChanges) {
      setNodes(updatedNodes);
    }
  }, [
    nodes,
    triggerCatalog,
    integrationContext,
    setNodes,
  ]);
}
