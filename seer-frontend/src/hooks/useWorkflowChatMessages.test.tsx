import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useChatStore, useUIStore } from '@/stores';
import type { ChatMessage } from '@/components/workflows/buildtypes';

import { useWorkflowChatMessages } from './useWorkflowChatMessages';
import { useChatMessages } from './useChatMessages';

vi.mock('./useChatMessages', () => ({
  useChatMessages: vi.fn(),
}));

vi.mock('@/lib/workflow-graph', () => ({
  workflowSpecToGraph: vi.fn(() => ({ nodes: [], edges: [] })),
}));

const mockUseChatMessages = vi.mocked(useChatMessages);
const localStorageMock = {
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  key: vi.fn(() => null),
  length: 0,
};

describe('useWorkflowChatMessages', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, 'localStorage', {
      value: localStorageMock,
      writable: true,
    });
    useChatStore.getState().resetChatState();
    useUIStore.getState().resetUIState();
  });

  it('uses persisted session messages from the query payload object', () => {
    const persistedMessages: ChatMessage[] = [
      {
        role: 'assistant',
        content: 'Persisted reply',
        timestamp: new Date('2026-03-18T00:00:00.000Z'),
      },
    ];

    mockUseChatMessages.mockReturnValue({
      data: {
        currentExecutionStatus: null,
        currentExecutionTaskId: null,
        pendingInterruptType: null,
        pendingInterruptData: null,
        messages: persistedMessages,
      },
    } as ReturnType<typeof useChatMessages>);

    const { result } = renderHook(() => useWorkflowChatMessages('wf_123'));

    expect(result.current.messages).toEqual(persistedMessages);
  });

  it('restores proposal preview from persisted pending proposals', async () => {
    const persistedMessages: ChatMessage[] = [
      {
        role: 'assistant',
        content: 'Proposal ready',
        timestamp: new Date('2026-03-18T00:00:00.000Z'),
        proposal: {
          id: 7,
          workflow_id: 'wf_123',
          created_by: { user_id: 'user_1' },
          summary: 'Add Gmail trigger',
          status: 'pending',
          spec: {
            version: '2',
            nodes: [],
            edges: [],
          },
          created_at: '2026-03-18T00:00:00.000Z',
          updated_at: '2026-03-18T00:00:00.000Z',
        },
      },
    ];

    mockUseChatMessages.mockReturnValue({
      data: {
        currentExecutionStatus: null,
        currentExecutionTaskId: null,
        pendingInterruptType: null,
        pendingInterruptData: null,
        messages: persistedMessages,
      },
    } as ReturnType<typeof useChatMessages>);

    renderHook(() => useWorkflowChatMessages('wf_123'));

    await waitFor(() => {
      expect(useUIStore.getState().proposalPreview?.proposal.id).toBe(7);
    });
  });

  it('deduplicates a transient assistant message when the persisted copy is slightly earlier', async () => {
    useChatStore.setState({
      currentSessionId: 153,
      transientSessionId: 153,
      transientMessages: [
        {
          role: 'assistant',
          content: "Nice to meet you! I see this is your first time here.",
          timestamp: new Date('2026-03-18T13:25:18.960Z'),
        },
      ],
    });

    const persistedMessages: ChatMessage[] = [
      {
        role: 'assistant',
        content: "Nice to meet you! I see this is your first time here.",
        thinking: ["Calling tool 'get_user_profile' with args: {}"],
        timestamp: new Date('2026-03-18T13:25:18.948Z'),
      },
    ];

    mockUseChatMessages.mockReturnValue({
      data: {
        currentExecutionStatus: 'completed',
        currentExecutionTaskId: null,
        pendingInterruptType: null,
        pendingInterruptData: null,
        messages: persistedMessages,
      },
    } as ReturnType<typeof useChatMessages>);

    const { result } = renderHook(() => useWorkflowChatMessages('wf_205'));

    await waitFor(() => {
      expect(result.current.messages).toEqual(persistedMessages);
      expect(useChatStore.getState().transientMessages).toEqual([]);
    });
  });
});
