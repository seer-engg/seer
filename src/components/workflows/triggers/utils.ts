import type { InputDef, JsonValue, TriggerSpec } from '@/types/workflow-spec';

export type BindingMode = 'event' | 'literal';

export interface BindingConfig {
  mode: BindingMode;
  value: string;
}

export interface BindingState {
  [inputName: string]: BindingConfig;
}

export interface GmailConfigState {
  labelIds: string;
  query: string;
  maxResults: string;
  overlapMs: string;
}

export interface CronConfigState {
  cronExpression: string;
  timezone: string;
  description: string;
}

export type SupabaseEventType = 'INSERT' | 'UPDATE' | 'DELETE';

export interface SupabaseConfigState {
  integrationResourceId: string;
  integrationResourceLabel?: string;
  schema: string;
  table: string;
  events: SupabaseEventType[];
}

const EVENT_PREFIX = 'event.';

export const makeDefaultGmailConfig = (): GmailConfigState => ({
  labelIds: 'INBOX',
  query: '',
  maxResults: '25',
  overlapMs: '300000',
});

export const SUPABASE_EVENT_TYPES: SupabaseEventType[] = ['INSERT', 'UPDATE', 'DELETE'];

export const makeDefaultSupabaseConfig = (): SupabaseConfigState => ({
  integrationResourceId: '',
  integrationResourceLabel: '',
  schema: 'public',
  table: '',
  events: [...SUPABASE_EVENT_TYPES],
});

export function buildDefaultBindingState(workflowInputs: Record<string, InputDef>): BindingState {
  return Object.keys(workflowInputs).reduce<BindingState>((acc, inputName) => {
    acc[inputName] = {
      mode: 'event',
      value: `data.${inputName}`,
    };
    return acc;
  }, {});
}

const isEventExpression = (value: unknown): value is string => {
  return typeof value === 'string' && value.trim().startsWith('${event.');
};

const stripEventExpression = (value: string): string => {
  const trimmed = value.trim();
  if (!trimmed.startsWith('${') || !trimmed.endsWith('}')) {
    return trimmed;
  }
  const inner = trimmed.slice(2, -1);
  if (inner.startsWith(EVENT_PREFIX)) {
    return inner.slice(EVENT_PREFIX.length);
  }
  return inner;
};

export function deriveBindingStateFromSubscription(
  workflowInputs: Record<string, InputDef>,
  subscription: TriggerSpec,
): BindingState {
  const defaults = buildDefaultBindingState(workflowInputs);
  const bindings = (subscription.filters ?? {}) as Record<string, unknown>;

  return Object.keys(workflowInputs).reduce<BindingState>((acc, inputName) => {
    const existing = bindings[inputName];
    if (isEventExpression(existing)) {
      acc[inputName] = {
        mode: 'event',
        value: stripEventExpression(existing),
      };
      return acc;
    }

    if (existing !== undefined && existing !== null) {
      acc[inputName] = {
        mode: 'literal',
        value: typeof existing === 'string' ? existing : JSON.stringify(existing),
      };
      return acc;
    }

    acc[inputName] = defaults[inputName] ?? { mode: 'event', value: '' };
    return acc;
  }, defaults);
}

export function sanitizeEventExpression(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) {
    return '';
  }
  if (trimmed.startsWith('${') && trimmed.endsWith('}')) {
    return trimmed;
  }
  const normalized = trimmed.startsWith(EVENT_PREFIX)
    ? trimmed
    : `${EVENT_PREFIX}${trimmed.replace(/^\.*/, '')}`;
  return `\${${normalized}}`;
}

const coerceLiteralValue = (rawValue: string): JsonValue => {
  const value = rawValue.trim();
  if (!value.length) {
    return '';
  }
  if (value === 'true') return true;
  if (value === 'false') return false;
  if (value === 'null') return null;
  const numeric = Number(value);
  if (!Number.isNaN(numeric) && value === numeric.toString()) {
    return numeric;
  }
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};

export function buildBindingsPayload(bindingState: BindingState): Record<string, JsonValue> {
  return Object.entries(bindingState).reduce<Record<string, JsonValue>>((acc, [inputName, config]) => {
    if (!config?.value) {
      return acc;
    }
    if (config.mode === 'event') {
      const expression = sanitizeEventExpression(config.value);
      if (expression) {
        acc[inputName] = expression;
      }
      return acc;
    }
    acc[inputName] = coerceLiteralValue(config.value);
    return acc;
  }, {});
}

export function buildDefaultBindings(workflowInputs: Record<string, InputDef>): Record<string, JsonValue> {
  const bindingState = buildDefaultBindingState(workflowInputs);
  return buildBindingsPayload(bindingState);
}

export function buildGmailConfigFromProviderConfig(
  providerConfig?: Record<string, unknown> | null,
): GmailConfigState {
  if (!providerConfig) {
    return makeDefaultGmailConfig();
  }
  const labelIdsValue = providerConfig['label_ids'];
  const labelIds = Array.isArray(labelIdsValue) ? labelIdsValue.join(', ') : providerConfig['label_ids'] ?? 'INBOX';
  return {
    labelIds: String(labelIds || 'INBOX'),
    query: String(providerConfig['query'] ?? ''),
    maxResults: String(providerConfig['max_results'] ?? '25'),
    overlapMs: String(providerConfig['overlap_ms'] ?? '300000'),
  };
}

export function serializeGmailConfig(state: GmailConfigState): Record<string, JsonValue> {
  const providerConfig: Record<string, JsonValue> = {};
  const labels = state.labelIds
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean);
  if (labels.length) {
    providerConfig.label_ids = labels;
  }
  const query = state.query.trim();
  if (query) {
    providerConfig.query = query;
  }
  const maxResultsValue = Number.parseInt(state.maxResults, 10);
  if (!Number.isNaN(maxResultsValue)) {
    providerConfig.max_results = Math.min(Math.max(maxResultsValue, 1), 25);
  }
  const overlapMsValue = Number.parseInt(state.overlapMs, 10);
  if (!Number.isNaN(overlapMsValue)) {
    providerConfig.overlap_ms = Math.min(Math.max(overlapMsValue, 0), 900000);
  }
  return providerConfig;
}

export function buildSupabaseConfigFromProviderConfig(
  providerConfig?: Record<string, unknown> | null,
): SupabaseConfigState {
  if (!providerConfig) {
    return makeDefaultSupabaseConfig();
  }
  const rawEvents = Array.isArray(providerConfig['events']) ? providerConfig['events'] : SUPABASE_EVENT_TYPES;
  const normalizedEvents = rawEvents
    .map((event) => String(event).toUpperCase() as SupabaseEventType)
    .filter((event): event is SupabaseEventType => SUPABASE_EVENT_TYPES.includes(event));

  return {
    integrationResourceId: providerConfig['integration_resource_id']
      ? String(providerConfig['integration_resource_id'])
      : '',
    integrationResourceLabel:
      typeof providerConfig['integration_resource_label'] === 'string'
        ? providerConfig['integration_resource_label']
        : '',
    schema: providerConfig['schema'] ? String(providerConfig['schema']) : 'public',
    table: providerConfig['table'] ? String(providerConfig['table']) : '',
    events: normalizedEvents.length ? normalizedEvents : [...SUPABASE_EVENT_TYPES],
  };
}

export function serializeSupabaseConfig(state: SupabaseConfigState): Record<string, JsonValue> {
  const payload: Record<string, JsonValue> = {};
  const resourceId = Number(state.integrationResourceId);
  if (!Number.isNaN(resourceId)) {
    payload.integration_resource_id = resourceId;
  }
  if (state.integrationResourceLabel) {
    payload.integration_resource_label = state.integrationResourceLabel;
  }
  const schema = state.schema.trim() || 'public';
  if (schema) {
    payload.schema = schema;
  }
  const table = state.table.trim();
  if (table) {
    payload.table = table;
  }
  const normalizedEvents = state.events.filter((event) => SUPABASE_EVENT_TYPES.includes(event));
  if (normalizedEvents.length) {
    payload.events = normalizedEvents;
  }
  return payload;
}

export function validateSupabaseConfig(state: SupabaseConfigState): {
  valid: boolean;
  errors: Partial<Record<'resource' | 'table' | 'events', string>>;
} {
  const errors: Partial<Record<'resource' | 'table' | 'events', string>> = {};
  if (!state.integrationResourceId) {
    errors.resource = 'Select a Supabase project';
  }
  if (!state.table.trim()) {
    errors.table = 'Table name is required';
  }
  if (!state.events.some((event) => SUPABASE_EVENT_TYPES.includes(event))) {
    errors.events = 'Select at least one event type';
  }
  return { valid: Object.keys(errors).length === 0, errors };
}

export const makeDefaultCronConfig = (): CronConfigState => ({
  cronExpression: '0 9 * * *',
  timezone: 'America/Chicago',
  description: '',
});

export function buildCronConfigFromProviderConfig(
  providerConfig?: Record<string, unknown> | null,
): CronConfigState {
  if (!providerConfig) {
    return makeDefaultCronConfig();
  }
  return {
    cronExpression: String(providerConfig['cron_expression'] ?? '0 9 * * *'),
    timezone: String(providerConfig['timezone'] ?? 'America/Chicago'),
    description: String(providerConfig['description'] ?? ''),
  };
}

export function serializeCronConfig(state: CronConfigState): Record<string, JsonValue> {
  const providerConfig: Record<string, JsonValue> = {
    cron_expression: state.cronExpression.trim(),
    timezone: state.timezone,
  };
  const description = state.description.trim();
  if (description) {
    providerConfig.description = description;
  }
  return providerConfig;
}

export function validateCronExpression(expression: string): { valid: boolean; error?: string } {
  const trimmed = expression.trim();
  if (!trimmed) {
    return { valid: false, error: 'Cron expression is required' };
  }

  const parts = trimmed.split(/\s+/);
  if (parts.length !== 5) {
    return { valid: false, error: 'Cron expression must have 5 fields (minute hour day month weekday)' };
  }

  const cronFieldRegex = /^(\*|[0-9]+(-[0-9]+)?(\/[0-9]+)?)(,(\*|[0-9]+(-[0-9]+)?(\/[0-9]+)?))*$/;
  for (let i = 0; i < parts.length; i++) {
    if (!cronFieldRegex.test(parts[i])) {
      return { valid: false, error: `Invalid syntax in field ${i + 1}: ${parts[i]}` };
    }
  }

  return { valid: true };
}
