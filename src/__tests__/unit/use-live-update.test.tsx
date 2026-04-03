import { useEffect, useRef } from 'react';
import { renderHook, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData } from '@/components/workflows/types';
import { useLiveUpdate } from '@/components/workflows/panels/hooks/useLiveUpdate';
import type { ToolBlockConfig } from '@/components/workflows/block-config/types';

function createToolNode(config: ToolBlockConfig): Node<WorkflowNodeData> {
  return {
    id: 'tool-1',
    type: 'tool',
    position: { x: 0, y: 0 },
    data: {
      type: 'tool',
      label: 'Tool',
      config,
    },
  } as Node<WorkflowNodeData>;
}

function useLiveUpdateHarness(params: {
  node: Node<WorkflowNodeData> | null;
  currentConfig: ToolBlockConfig;
  onUpdate: ReturnType<typeof vi.fn>;
}) {
  const configRef = useRef(params.currentConfig);

  useEffect(() => {
    configRef.current = params.currentConfig;
  }, [params.currentConfig]);

  useLiveUpdate({
    enabled: true,
    delayMs: 350,
    node: params.node,
    currentConfig: params.currentConfig,
    currentInputRefs: {},
    currentOauthScope: undefined,
    configRef,
    onUpdate: params.onUpdate,
  });
}

describe('useLiveUpdate', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does not save when only local-only config fields changed', async () => {
    const onUpdate = vi.fn();
    const node = createToolNode({
      tool_name: 'gmail.send',
      params: { to: 'user@example.com' },
    });

    renderHook(() =>
      useLiveUpdateHarness({
        node,
        currentConfig: {
          tool_name: 'gmail.send',
          params: { to: 'user@example.com' },
          integration_type: 'gmail',
          provider: 'google',
        },
        onUpdate,
      }),
    );

    await act(async () => {
      vi.advanceTimersByTime(400);
    });

    expect(onUpdate).not.toHaveBeenCalled();
  });

  it('debounces and saves real persisted edits', async () => {
    const onUpdate = vi.fn();
    const node = createToolNode({
      tool_name: 'gmail.send',
      params: { to: 'user@example.com' },
    });

    renderHook(() =>
      useLiveUpdateHarness({
        node,
        currentConfig: {
          tool_name: 'gmail.send',
          params: { to: 'team@example.com' },
          integration_type: 'gmail',
        },
        onUpdate,
      }),
    );

    await act(async () => {
      vi.advanceTimersByTime(400);
    });

    expect(onUpdate).toHaveBeenCalledTimes(1);
    expect(onUpdate).toHaveBeenCalledWith('tool-1', {
      config: {
        tool_name: 'gmail.send',
        params: { to: 'team@example.com' },
      },
    });
  });
});
