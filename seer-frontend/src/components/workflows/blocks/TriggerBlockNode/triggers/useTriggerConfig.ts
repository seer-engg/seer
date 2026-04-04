/**
 * Hook for managing trigger configuration with autosave.
 * Uses direct store updates instead of draft system.
 */

import { useCallback, useContext, useMemo } from 'react';
import { toast } from '@/components/ui/sonner';
import { useActiveWorkflowId } from '@/hooks/useActiveWorkflowId';
import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import { WorkflowCanvasContext } from '../../../canvas/workflow-canvas-context';
import type { JsonObject } from '@/types/workflow-spec';
import type { TriggerDescriptor } from '@/types/triggers';
import type { WorkflowNodeData } from '../../../types';
import { useDynamicTriggerConfig } from './useDynamicTriggerConfig';
import { useTriggersStore } from '@/stores/triggersStore';
import { useCanvasStore } from '@/stores';
import {
  deriveQuickOptionsFromSchema,
  deriveQuickOptionsFromUserSchema,
  hasFlexibleDataSchema,
  createDefaultUserSchema,
  type QuickOption,
} from '../../../triggers/schema-utils';
import {
  buildSupabaseConfigFromProviderConfig,
  SUPABASE_EVENT_TYPES,
  validateSupabaseConfig,
} from '../../../triggers/utils';

// Trigger kinds for special handling
export type TriggerKind = 'gmail' | 'cron' | 'supabase' | 'webhook' | 'form' | 'discord' | 'slack' | 'calendar' | 'airtable' | 'google_sheets' | 'meetings_ea';

// Constants for trigger key detection
const WEBHOOK_TRIGGER_KEY = 'webhook.generic';
const FORM_TRIGGER_KEY = 'form.hosted';

type TriggerSubscription = NonNullable<WorkflowNodeData['triggerMeta']>['subscription'];

/**
 * Extracts data schema from event_schema.properties.data.
 * Returns a default user schema if event_schema is not present.
 */
function extractDataSchema(subscription: TriggerSubscription): JsonObject {
  const eventSchema = subscription?.event_schema;

  if (!eventSchema) {
    return createDefaultUserSchema();
  }

  const properties = (eventSchema as JsonObject).properties as JsonObject | undefined;
  const dataSchema = properties?.data as JsonObject | undefined;

  return dataSchema ?? createDefaultUserSchema();
}

/**
 * Builds a complete event schema with the user's data schema embedded.
 * For webhook/form triggers, this creates the full envelope structure.
 */
function buildEventSchemaWithData(
  descriptor: TriggerDescriptor | null | undefined,
  dataSchema: JsonObject,
): JsonObject {
  // If descriptor has an event_schema template, use it and merge data
  if (descriptor?.event_schema) {
    const baseSchema = descriptor.event_schema as JsonObject;
    const properties = (baseSchema.properties as JsonObject) ?? {};

    return {
      ...baseSchema,
      properties: {
        ...properties,
        data: dataSchema,
      },
    };
  }

  // Default webhook envelope structure
  return {
    type: 'object',
    properties: {
      headers: {
        type: 'object',
        additionalProperties: { type: 'string' },
      },
      query: {
        type: 'object',
        additionalProperties: { type: 'string' },
      },
      data: dataSchema,
    },
    required: ['data'],
  };
}

/**
 * Determines the trigger kind from its key.
 */
function keyToKind(triggerKey: string): TriggerKind {
  if (triggerKey === WEBHOOK_TRIGGER_KEY) return 'webhook';
  if (triggerKey === FORM_TRIGGER_KEY) return 'form';
  if (triggerKey.includes('gmail')) return 'gmail';
  if (triggerKey.includes('cron')) return 'cron';
  if (triggerKey.includes('supabase')) return 'supabase';
  if (triggerKey.includes('discord')) return 'discord';
  if (triggerKey.includes('slack')) return 'slack';
  if (triggerKey.includes('calendar') || triggerKey.includes('google_calendar')) return 'calendar';
  if (triggerKey.includes('airtable')) return 'airtable';
  if (triggerKey.includes('google_sheets')) return 'google_sheets';
  if (triggerKey.includes('meetings_ea')) return 'meetings_ea';
  return 'webhook';
}

/**
 * Checks if a trigger supports user-defined schemas.
 */
function supportsUserSchema(triggerKey: string, descriptor?: TriggerDescriptor | null): boolean {
  const kind = keyToKind(triggerKey);
  if (kind === 'webhook' || kind === 'form') {
    return true;
  }
  // Also check if event_schema has flexible data
  if (descriptor?.event_schema) {
    return hasFlexibleDataSchema(descriptor.event_schema);
  }
  return false;
}

/**
 * Checks if a config schema has actual properties to render.
 * Empty schemas (no properties) should not trigger dynamic config mode.
 */
function hasUsableConfigSchema(schema: JsonObject | null | undefined): boolean {
  if (!schema) return false;
  const properties = schema.properties as Record<string, unknown> | undefined;
  return !!properties && Object.keys(properties).length > 0;
}

/**
 * Resolves the effective trigger key from available sources.
 */
function resolveTriggerKey(
  triggerKey: string,
  subscription: TriggerSubscription,
  descriptor: TriggerDescriptor | null,
): string {
  return triggerKey || subscription?.key || descriptor?.key || '';
}

/**
 * Derives quick options based on trigger type.
 */
function deriveQuickOptions(
  isUserSchemaSupported: boolean,
  descriptor: TriggerDescriptor | null,
  dataSchema: JsonObject,
): QuickOption[] {
  if (isUserSchemaSupported && !hasUsableConfigSchema(descriptor?.config_schema)) {
    return dataSchema ? deriveQuickOptionsFromUserSchema(dataSchema) : [];
  }
  return descriptor?.event_schema ? deriveQuickOptionsFromSchema(descriptor.event_schema) : [];
}

/**
 * Serializes dynamic config values, filtering out empty optional fields.
 */
function serializeConfigValues(
  configValues: Record<string, unknown>,
  key: string,
  value: unknown,
  requiredFields: string[],
): JsonObject {
  const updated = { ...configValues, [key]: value };
  const serialized: JsonObject = {};

  for (const [configKey, configValue] of Object.entries(updated)) {
    if (configValue === undefined || configValue === null) continue;
    if (configValue === '' && !requiredFields.includes(configKey)) continue;
    serialized[configKey] = configValue as JsonObject[string];
  }

  return serialized;
}

/**
 * Builds and validates Supabase provider_config, returning null if invalid.
 */
function buildSupabaseProviderConfig(
  baseConfig: JsonObject,
  configValues: Record<string, unknown>,
  key: string,
  value: unknown,
): { providerConfig: JsonObject; resourceId: number } | { error: string } {
  const normalizedEvents =
    Array.isArray(baseConfig.events) && baseConfig.events.length > 0
      ? baseConfig.events
      : [...SUPABASE_EVENT_TYPES];

  const configWithEvents = { ...baseConfig, events: normalizedEvents };
  const supabaseConfig = buildSupabaseConfigFromProviderConfig({ ...configValues, [key]: value });
  const validation = validateSupabaseConfig(supabaseConfig);

  if (!validation.valid) {
    return { error: Object.values(validation.errors).find(Boolean) ?? 'Configuration incomplete' };
  }

  const resourceId = Number(supabaseConfig.integrationResourceId);
  if (Number.isNaN(resourceId)) {
    return { error: 'Select a Supabase project.' };
  }

  return {
    resourceId,
    providerConfig: {
      ...configWithEvents,
      integration_resource_id: resourceId,
      ...(supabaseConfig.integrationResourceLabel
        ? { integration_resource_label: supabaseConfig.integrationResourceLabel }
        : {}),
      schema: supabaseConfig.schema || 'public',
      table: supabaseConfig.table,
      events: supabaseConfig.events,
    },
  };
}

export type TriggerConfigState =
  | {
      kind: 'webhook';
      dataSchema: JsonObject;
      setDataSchema: (schema: JsonObject) => void;
    }
  | {
      kind: 'dynamic';
      configValues: Record<string, unknown>;
      setConfigValue: (key: string, value: unknown) => void;
    };

export const useTriggerConfig = (
  triggerId: string,
  triggerKey: string,
  triggerMeta: NonNullable<WorkflowNodeData['triggerMeta']>,
  options?: { readOnly?: boolean },
) => {
  const workflowId = useActiveWorkflowId();
  const canvasContext = useContext(WorkflowCanvasContext);
  const readOnly = options?.readOnly ?? canvasContext?.readOnly ?? false;
  const { data: triggerCatalog = [] } = useTriggerCatalogQuery({ enabled: !readOnly });

  // Read subscription from store to get fresh provider_config after edits.
  // Falls back to triggerMeta.subscription for unsaved workflows or if not found in store.
  const subscription = useTriggersStore((state) => {
    if (!workflowId) return triggerMeta.subscription;
    const triggers = state.workflowTriggers.get(workflowId) ?? [];
    return triggers.find((t) => t.id === triggerId) ?? triggerMeta.subscription;
  });

  const descriptor = useMemo(
    () => triggerMeta.descriptor ?? triggerCatalog.find((entry) => entry.key === triggerKey) ?? null,
    [triggerCatalog, triggerKey, triggerMeta.descriptor],
  );
  const resolvedTriggerKey = resolveTriggerKey(triggerKey, subscription, descriptor);

  const kind = keyToKind(resolvedTriggerKey);
  const isUserSchemaSupported = supportsUserSchema(resolvedTriggerKey, descriptor);

  // Extract data schema from schemas.event.properties.data
  const dataSchema = useMemo(() => extractDataSchema(subscription), [subscription]);

  // Update data schema with autosave
  const setDataSchema = useCallback(
    (newSchema: JsonObject) => {
      if (!workflowId) {
        toast.error('Unable to save', { description: 'Save the workflow first' });
        return;
      }

      if (!subscription?.id) {
        toast.error('Unable to save', { description: 'Trigger not initialized' });
        return;
      }

      const fullEventSchema = buildEventSchemaWithData(descriptor, newSchema);

      useTriggersStore.getState().updateTrigger(workflowId, subscription.id, {
        event_schema: fullEventSchema,
      });

      useCanvasStore.getState().markDirty(); // Triggers debounced autosave
    },
    [workflowId, subscription, descriptor],
  );

  // Dynamic config for other trigger types
  const dynamicConfig = useDynamicTriggerConfig(descriptor?.config_schema, subscription);

  // Persist a dynamic config field change to the store
  const setConfigValue = useCallback(
    (key: string, value: unknown) => {
      dynamicConfig.setConfigValue(key, value);

      if (!workflowId) {
        toast.error('Unable to save', { description: 'Save the workflow first' });
        return;
      }
      if (!subscription?.id) {
        toast.error('Unable to save', { description: 'Trigger not initialized' });
        return;
      }

      if (kind === 'supabase') {
        const result = buildSupabaseProviderConfig(
          dynamicConfig.serializeConfig(), dynamicConfig.configValues, key, value,
        );
        if ('error' in result) {
          toast.error('Supabase configuration incomplete', { description: result.error });
          return;
        }
        useTriggersStore.getState().updateTrigger(workflowId, subscription.id, {
          provider_config: result.providerConfig,
        });
      } else {
        const required = (descriptor?.config_schema?.required as string[]) || [];
        const serialized = serializeConfigValues(dynamicConfig.configValues, key, value, required);
        useTriggersStore.getState().updateTrigger(workflowId, subscription.id, {
          provider_config: serialized,
        });
      }

      useCanvasStore.getState().markDirty();
    },
    [dynamicConfig, workflowId, subscription?.id, kind, descriptor?.config_schema],
  );

  // Build the config state based on trigger type
  const state: TriggerConfigState =
    isUserSchemaSupported && !hasUsableConfigSchema(descriptor?.config_schema)
      ? { kind: 'webhook', dataSchema, setDataSchema }
      : { kind: 'dynamic', configValues: dynamicConfig.configValues, setConfigValue };

  const quickOptions = useMemo(
    () => deriveQuickOptions(isUserSchemaSupported, descriptor, dataSchema),
    [isUserSchemaSupported, descriptor, dataSchema],
  );

  const isLoading = !triggerMeta.handlers;

  return {
    kind,
    state,
    quickOptions,
    isLoading,
    // Expose for special handling (e.g., details sections)
    isUserSchemaSupported,
    descriptor,
    resolvedTriggerKey,
    subscription,
  } as const;
};
