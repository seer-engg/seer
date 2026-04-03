import type { ToolBlockConfig } from '@/components/workflows/block-config/types';
import type { WorkflowNodeData } from '../../types';

const LOCAL_ONLY_CONFIG_KEYS = new Set([
  '__resourceLabels',
  'input_refs',
  'integration_type',
  'provider',
]);

export function sanitizeConfigForPersistence(config: ToolBlockConfig = {}): ToolBlockConfig {
  return Object.fromEntries(
    Object.entries(config).filter(([key]) => !LOCAL_ONLY_CONFIG_KEYS.has(key)),
  ) as ToolBlockConfig;
}

export function buildPersistedConfig(
  originalConfig: ToolBlockConfig,
  currentConfig: ToolBlockConfig,
): ToolBlockConfig {
  const mergedConfig: ToolBlockConfig = {
    ...sanitizeConfigForPersistence(originalConfig),
    ...sanitizeConfigForPersistence(currentConfig),
  };

  if ('fields' in currentConfig) {
    mergedConfig.fields = currentConfig.fields;
  }

  if ('output_schema' in currentConfig) {
    mergedConfig.output_schema = currentConfig.output_schema;
  }

  return mergedConfig;
}

export function hasPersistedConfigChanges(params: {
  currentConfig: ToolBlockConfig;
  originalConfig: ToolBlockConfig;
}): boolean {
  const nextConfig = buildPersistedConfig(params.originalConfig, params.currentConfig);
  const previousConfig = sanitizeConfigForPersistence(params.originalConfig);
  return JSON.stringify(nextConfig) !== JSON.stringify(previousConfig);
}

export function buildPersistedNodeUpdate(params: {
  originalConfig: ToolBlockConfig;
  currentConfig: ToolBlockConfig;
}): Partial<WorkflowNodeData> | null {
  if (!hasPersistedConfigChanges(params)) {
    return null;
  }

  return {
    config: buildPersistedConfig(params.originalConfig, params.currentConfig),
  };
}
