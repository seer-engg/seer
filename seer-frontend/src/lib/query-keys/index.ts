import type { FileListParams } from '@/lib/files-api';
import type {
  AnalyticsParams,
  RecordsParams,
  WorkflowAnalyticsParams,
} from '@/lib/usage-analytics-api';
import type {
  EmailEventsParams,
  EmailSummaryParams,
} from '@/lib/email-analytics-api';
import type { TriggerSubscriptionFilters } from '@/types/triggers';

type Serializable =
  | boolean
  | null
  | number
  | string
  | undefined
  | Serializable[]
  | { [key: string]: Serializable };

const EMPTY_OBJECT = Object.freeze({}) as Readonly<Record<string, never>>;

function normalizeValue(value: Serializable): Serializable {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeValue(item));
  }

  if (value && typeof value === 'object') {
    return normalizeObject(value);
  }

  return value;
}

function normalizeObject<T extends Record<string, Serializable>>(
  params?: T,
): Readonly<Partial<T>> {
  if (!params) {
    return EMPTY_OBJECT as Readonly<Partial<T>>;
  }

  const entries = Object.entries(params)
    .filter(([, value]) => value !== undefined && value !== '')
    .map(([key, value]) => [key, normalizeValue(value)])
    .sort(([left], [right]) => left.localeCompare(right));

  if (entries.length === 0) {
    return EMPTY_OBJECT as Readonly<Partial<T>>;
  }

  return Object.freeze(Object.fromEntries(entries)) as Readonly<Partial<T>>;
}

export const userKeys = {
  all: ['user'] as const,
  settings: () => [...userKeys.all, 'settings'] as const,
};

export const memoryKeys = {
  all: ['memory'] as const,
  stats: () => [...memoryKeys.all, 'stats'] as const,
  banks: () => [...memoryKeys.all, 'banks'] as const,
  bank: (memoryBankId: string | null | undefined) =>
    [...memoryKeys.banks(), memoryBankId ?? null] as const,
  lists: () => [...memoryKeys.all, 'list'] as const,
  list: (query?: string | null) => {
    const normalizedQuery = query?.trim();
    const normalizedParams = normalizeObject({
      query: normalizedQuery || undefined,
    });

    return Object.keys(normalizedParams).length === 0
      ? memoryKeys.lists()
      : [...memoryKeys.lists(), normalizedParams] as const;
  },
  items: (memoryBankId: string | null | undefined, query?: string | null) => {
    const normalizedQuery = query?.trim();
    const normalizedParams = normalizeObject({
      query: normalizedQuery || undefined,
    });
    const baseKey = [...memoryKeys.all, 'items', memoryBankId ?? null] as const;

    return Object.keys(normalizedParams).length === 0
      ? baseKey
      : [...baseKey, normalizedParams] as const;
  },
};

export const workflowRunKeys = {
  all: ['workflow-runs'] as const,
  lists: () => workflowRunKeys.all,
  list: (workflowId: string | null | undefined) =>
    [...workflowRunKeys.all, workflowId ?? null] as const,
};

export const workflowKeys = {
  all: ['workflows'] as const,
  lists: () => [...workflowKeys.all, 'list'] as const,
  list: () => workflowKeys.lists(),
  details: () => [...workflowKeys.all, 'detail'] as const,
  detail: (workflowId: string | null | undefined) =>
    [...workflowKeys.details(), workflowId ?? null] as const,
  versions: () => [...workflowKeys.all, 'versions'] as const,
  versionList: (workflowId: string | null | undefined) =>
    [...workflowKeys.versions(), workflowId ?? null] as const,
};

export const runHistoryKeys = {
  all: ['run-history'] as const,
  detail: (runId: string | undefined) => [...runHistoryKeys.all, runId ?? null] as const,
};

export const runInterruptKeys = {
  all: ['run-interrupt'] as const,
  detail: (runId: string | undefined) => [...runInterruptKeys.all, runId ?? null] as const,
};

export const chatKeys = {
  all: ['chat'] as const,
  sessions: () => [...chatKeys.all, 'sessions'] as const,
  sessionsByWorkflow: (workflowId: string | null) =>
    [...chatKeys.sessions(), workflowId] as const,
  messages: () => [...chatKeys.all, 'messages'] as const,
  messagesBySession: (sessionId: number | null) =>
    [...chatKeys.messages(), sessionId] as const,
};

export const triggerKeys = {
  all: ['triggers'] as const,
  catalog: () => [...triggerKeys.all, 'catalog'] as const,
  subscriptions: () => [...triggerKeys.all, 'subscriptions'] as const,
  subscriptionList: (filters: TriggerSubscriptionFilters = {}) =>
    [...triggerKeys.subscriptions(), 'list', normalizeObject(filters)] as const,
  subscriptionEnabled: () => [...triggerKeys.all, 'subscription-enabled'] as const,
  subscriptionEnabledDetail: (subscriptionId: number | null) =>
    [...triggerKeys.subscriptionEnabled(), subscriptionId] as const,
  events: () => [...triggerKeys.all, 'events'] as const,
  eventList: ({
    filterParams,
    provider,
    providerConnectionId,
    subscriptionId,
    triggerKey,
  }: {
    filterParams?: Record<string, Serializable>;
    provider: string;
    providerConnectionId?: number;
    subscriptionId?: number;
    triggerKey: string;
  }) =>
    [
      ...triggerKeys.events(),
      'list',
      normalizeObject({
        filterParams: filterParams ? normalizeObject(filterParams) : undefined,
        provider,
        providerConnectionId,
        subscriptionId,
        triggerKey,
      }),
    ] as const,
};

export const fileKeys = {
  all: ['files'] as const,
  lists: () => [...fileKeys.all, 'list'] as const,
  list: (params: FileListParams = {}) =>
    [...fileKeys.lists(), normalizeObject(params)] as const,
  pickerLists: () => [...fileKeys.all, 'picker'] as const,
  pickerList: (params: FileListParams = {}) =>
    [...fileKeys.pickerLists(), normalizeObject(params)] as const,
  stats: () => [...fileKeys.all, 'stats'] as const,
  detail: (fileId: string | null | undefined) => [...fileKeys.all, 'detail', fileId ?? null] as const,
};

export const knowledgeKeys = {
  all: ['knowledge-bases'] as const,
  lists: () => knowledgeKeys.all,
  detail: (knowledgeBaseId: string | null | undefined) =>
    [...knowledgeKeys.all, 'detail', knowledgeBaseId ?? null] as const,
  documents: () => [...knowledgeKeys.all, 'documents'] as const,
  documentsByKnowledgeBase: (knowledgeBaseId: string | null | undefined) =>
    [...knowledgeKeys.documents(), knowledgeBaseId ?? null] as const,
};

export const browserKeys = {
  all: ['browser'] as const,
  profiles: () => [...browserKeys.all, 'profiles'] as const,
  recordings: () => [...browserKeys.all, 'recordings'] as const,
  recordingList: (params: {
    enabled?: boolean;
    limit?: number;
    offset?: number;
    profileId?: string;
    workflowRunId?: string;
  }) =>
    [...browserKeys.recordings(), 'list', normalizeObject(params)] as const,
  recordingDetail: (recordingId: string | null) =>
    [...browserKeys.recordings(), 'detail', recordingId] as const,
  recordingEvents: (recordingId: string | null, isPublic: boolean) =>
    [...browserKeys.recordings(), 'events', recordingId, normalizeObject({ public: isPublic })] as const,
};

export const modelKeys = {
  all: ['models'] as const,
  available: () => [...modelKeys.all, 'available'] as const,
  availableImages: () => [...modelKeys.all, 'available-images'] as const,
};

export const connectionKeys = {
  all: ['connections'] as const,
  list: () => [...connectionKeys.all, 'list'] as const,
  grouped: () => [...connectionKeys.all, 'grouped'] as const,
};

export const toolKeys = {
  all: ['tools'] as const,
  catalog: () => [...toolKeys.all, 'catalog'] as const,
  status: () => [...toolKeys.all, 'status'] as const,
  functionBlocks: () => [...toolKeys.all, 'function-blocks'] as const,
};

export const analyticsKeys = {
  all: ['usage-analytics'] as const,
  summary: (params: AnalyticsParams = {}) =>
    [...analyticsKeys.all, 'summary', normalizeObject(params)] as const,
  workflows: (params: WorkflowAnalyticsParams = {}) =>
    [...analyticsKeys.all, 'workflows', normalizeObject(params)] as const,
  records: (params: RecordsParams = {}) =>
    [...analyticsKeys.all, 'records', normalizeObject(params)] as const,
};

export const resourceKeys = {
  all: ['resources'] as const,
  slackChannels: () => [...resourceKeys.all, 'slack', 'channel'] as const,
  slackChannelsByWorkspace: (workspaceId: string | null | undefined) =>
    [...resourceKeys.slackChannels(), workspaceId ?? null] as const,
  browseList: (params: {
    currentParentId?: string;
    dependsOnParam?: string;
    normalizedEndpoint?: string | null;
    resourceType: string;
    searchQuery?: string;
  }) =>
    [...resourceKeys.all, 'browse', normalizeObject(params)] as const,
  supabaseProjects: (searchQuery = '') =>
    [...resourceKeys.all, 'supabase-projects', searchQuery] as const,
};

export const emailAnalyticsKeys = {
  all: ['email-analytics'] as const,
  events: (params: EmailEventsParams = {}) =>
    [...emailAnalyticsKeys.all, 'events', normalizeObject(params)] as const,
  summary: (params: EmailSummaryParams = {}) =>
    [...emailAnalyticsKeys.all, 'summary', normalizeObject(params)] as const,
};

export const meetingKeys = {
  all: ['meetings'] as const,
  list: () => [...meetingKeys.all, 'list'] as const,
  detail: (botId: string | null) => [...meetingKeys.all, 'detail', botId] as const,
  transcript: (botId: string | null) => [...meetingKeys.all, 'transcript', botId] as const,
  recording: (botId: string | null) => [...meetingKeys.all, 'recording', botId] as const,
};

export const postgresKeys = {
  all: ['postgres'] as const,
  bindings: () => [...postgresKeys.all, 'bindings'] as const,
};
