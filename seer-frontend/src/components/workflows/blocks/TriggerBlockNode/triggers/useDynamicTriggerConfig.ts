/**
 * Hook for managing dynamic trigger configuration state.
 * Replaces trigger-specific hooks (useGmailConfig, useCronConfig, useSupabaseConfig).
 */

import { useCallback, useEffect, useState } from 'react';
import type { JsonObject, JsonValue, TriggerSpec } from '@/types/workflow-spec';

/**
 * Extracts default values from a config_schema.
 */
function extractDefaultsFromSchema(
  schema: JsonObject | null | undefined,
): Record<string, unknown> {
  if (!schema) {
    return {};
  }

  const properties = schema.properties as Record<string, JsonObject> | undefined;
  if (!properties || typeof properties !== 'object') {
    return {};
  }

  const defaults: Record<string, unknown> = {};
  for (const [key, propSchema] of Object.entries(properties)) {
    if (propSchema && typeof propSchema === 'object' && 'default' in propSchema) {
      defaults[key] = propSchema.default;
    }
  }

  return defaults;
}

/**
 * Builds initial config values from subscription or schema defaults.
 */
function buildInitialConfigValues(
  configSchema: JsonObject | null | undefined,
  subscription: TriggerSpec | undefined,
): Record<string, unknown> {
  const defaults = extractDefaultsFromSchema(configSchema);

  // Priority: subscription > defaults
  if (subscription?.provider_config) {
    // Merge defaults with subscription config (subscription takes precedence)
    return { ...defaults, ...subscription.provider_config };
  }

  return defaults;
}

// Type coercer functions extracted to reduce complexity
const coerceToInteger = (value: unknown, defaultValue: JsonValue): JsonValue => {
  const numValue = typeof value === 'string' ? parseInt(value, 10) : value;
  return Number.isNaN(numValue) ? defaultValue ?? 0 : (numValue as JsonValue);
};

const coerceToNumber = (value: unknown, defaultValue: JsonValue): JsonValue => {
  const numValue = typeof value === 'string' ? parseFloat(value) : value;
  return Number.isNaN(numValue) ? defaultValue ?? 0 : (numValue as JsonValue);
};

const coerceToBoolean = (value: unknown): JsonValue => {
  if (typeof value === 'string') {
    return value === 'true';
  }
  return Boolean(value);
};

const coerceToArray = (value: unknown): JsonValue => {
  if (Array.isArray(value)) {
    return value as JsonValue;
  }
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
};

const coerceToObject = (value: unknown): JsonValue => {
  if (typeof value === 'object' && value !== null) {
    return value as JsonValue;
  }
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value);
      return typeof parsed === 'object' ? parsed : {};
    } catch {
      return {};
    }
  }
  return {};
};

/**
 * Coerces a value to the appropriate type based on the schema definition.
 */
function coerceValueForSchema(
  value: unknown,
  schema: JsonObject | null | undefined,
  key: string,
): JsonValue {
  if (!schema || !schema.properties) {
    return value as JsonValue;
  }

  const properties = schema.properties as Record<string, JsonObject>;
  const propSchema = properties[key];
  if (!propSchema) {
    return value as JsonValue;
  }

  const schemaType = propSchema.type as string | string[] | undefined;
  const primaryType = Array.isArray(schemaType) ? schemaType[0] : schemaType;
  const defaultValue = propSchema.default as JsonValue;

  switch (primaryType) {
    case 'integer':
      return coerceToInteger(value, defaultValue);
    case 'number':
      return coerceToNumber(value, defaultValue);
    case 'boolean':
      return coerceToBoolean(value);
    case 'array':
      return coerceToArray(value);
    case 'object':
      return coerceToObject(value);
    default:
      return String(value ?? '');
  }
}

export interface UseDynamicTriggerConfigResult {
  configValues: Record<string, unknown>;
  setConfigValue: (key: string, value: unknown) => void;
  setConfigValues: (values: Record<string, unknown>) => void;
  resetConfig: () => void;
  serializeConfig: () => Record<string, JsonValue>;
}

export function useDynamicTriggerConfig(
  configSchema: JsonObject | null | undefined,
  subscription?: TriggerSpec,
): UseDynamicTriggerConfigResult {
  const [configValues, setConfigValuesState] = useState<Record<string, unknown>>(() =>
    buildInitialConfigValues(configSchema, subscription),
  );

  // Sync when subscription changes
  useEffect(() => {
    setConfigValuesState(buildInitialConfigValues(configSchema, subscription));
  }, [configSchema, subscription]);

  const setConfigValue = useCallback(
    (key: string, value: unknown) => {
      setConfigValuesState((prev) => ({
        ...prev,
        [key]: value,
      }));
    },
    [],
  );

  const setConfigValues = useCallback((values: Record<string, unknown>) => {
    setConfigValuesState((prev) => ({
      ...prev,
      ...values,
    }));
  }, []);

  const resetConfig = useCallback(() => {
    setConfigValuesState(buildInitialConfigValues(configSchema, subscription));
  }, [configSchema, subscription]);

  const serializeConfig = useCallback((): Record<string, JsonValue> => {
    const serialized: Record<string, JsonValue> = {};

    for (const [key, value] of Object.entries(configValues)) {
      // Skip undefined/null values unless they're explicitly set
      if (value === undefined || value === null) {
        continue;
      }

      // Skip empty strings for optional fields
      if (value === '' && !isRequiredField(configSchema, key)) {
        continue;
      }

      serialized[key] = coerceValueForSchema(value, configSchema, key);
    }

    return serialized;
  }, [configValues, configSchema]);

  return {
    configValues,
    setConfigValue,
    setConfigValues,
    resetConfig,
    serializeConfig,
  };
}

/**
 * Checks if a field is required in the schema.
 */
function isRequiredField(
  schema: JsonObject | null | undefined,
  fieldName: string,
): boolean {
  if (!schema) {
    return false;
  }

  const required = schema.required as string[] | undefined;
  return Array.isArray(required) && required.includes(fieldName);
}
