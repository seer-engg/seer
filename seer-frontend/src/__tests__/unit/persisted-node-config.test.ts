import { describe, expect, it } from 'vitest';
import type { ToolBlockConfig } from '@/components/workflows/block-config/types';
import {
  buildPersistedConfig,
  buildPersistedNodeUpdate,
  sanitizeConfigForPersistence,
} from '@/components/workflows/panels/hooks/persistedNodeConfig';

describe('persistedNodeConfig', () => {
  it('removes local-only config keys from persisted payloads', () => {
    const config: ToolBlockConfig = {
      tool_name: 'gmail.send',
      integration_type: 'gmail',
      provider: 'google',
      input_refs: { subject: 'trigger.subject' },
      __resourceLabels: { thread_id: 'Inbox' },
      params: { to: 'user@example.com' },
    };

    expect(sanitizeConfigForPersistence(config)).toEqual({
      tool_name: 'gmail.send',
      params: { to: 'user@example.com' },
    });
  });

  it('does not emit an update when only local-only fields changed', () => {
    const originalConfig: ToolBlockConfig = {
      tool_name: 'gmail.send',
      params: { to: 'user@example.com' },
    };
    const currentConfig: ToolBlockConfig = {
      ...originalConfig,
      integration_type: 'gmail',
      provider: 'google',
      input_refs: { to: 'trigger.sender' },
    };

    expect(buildPersistedNodeUpdate({ originalConfig, currentConfig })).toBeNull();
  });

  it('preserves real persisted edits while stripping local-only fields', () => {
    const originalConfig: ToolBlockConfig = {
      tool_name: 'gmail.send',
      params: { to: 'user@example.com' },
    };
    const currentConfig: ToolBlockConfig = {
      ...originalConfig,
      integration_type: 'gmail',
      params: { to: 'team@example.com' },
      output_schema: { type: 'object', properties: {} },
    };

    expect(buildPersistedConfig(originalConfig, currentConfig)).toEqual({
      tool_name: 'gmail.send',
      params: { to: 'team@example.com' },
      output_schema: { type: 'object', properties: {} },
    });
  });
});
