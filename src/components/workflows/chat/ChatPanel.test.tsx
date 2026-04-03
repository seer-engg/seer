import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { useChatStore } from '@/stores/chatStore';

import { ChatPanel } from './ChatPanel';

const localStorageMock = {
  getItem: vi.fn(() => null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
  key: vi.fn(() => null),
  length: 0,
};

function renderChatPanel(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe('ChatPanel', () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();
    Object.defineProperty(window, 'localStorage', {
      value: localStorageMock,
      writable: true,
    });
    useChatStore.getState().resetChatState();
    useChatStore.setState({
      input: 'Use Gmail for invoices only',
      pendingInterruptData: {
        type: 'clarification_questions',
        clarification_questions: { questions: [] },
      },
    });
  });

  it('routes composer sends to resume handler while clarification is pending', () => {
    const onSend = vi.fn();
    const onResumeSend = vi.fn();

    renderChatPanel(
      <ChatPanel
        workflowId="wf_123"
        onSend={onSend}
        onResumeSend={onResumeSend}
        models={[]}
        isLoadingModels={false}
        filterSystemPrompt={(content) => content}
        onAcceptProposal={vi.fn()}
        onRejectProposal={vi.fn()}
      />,
    );

    fireEvent.click(screen.getAllByRole('button').at(-1)!);

    expect(onResumeSend).toHaveBeenCalledTimes(1);
    expect(onSend).not.toHaveBeenCalled();
  });

  it('falls back to the latest assistant interrupt message when store interrupt state is missing', () => {
    useChatStore.setState({
      pendingInterruptData: null,
      transientMessages: [
        {
          role: 'assistant',
          content: 'Which Gmail workflow would you like to create?',
          timestamp: new Date(),
          interruptRequired: true,
          interruptData: {
            type: 'clarification_questions',
            clarification_questions: { questions: [] },
          },
        },
      ],
    });

    const onSend = vi.fn();
    const onResumeSend = vi.fn();

    renderChatPanel(
      <ChatPanel
        workflowId="wf_123"
        onSend={onSend}
        onResumeSend={onResumeSend}
        models={[]}
        isLoadingModels={false}
        filterSystemPrompt={(content) => content}
        onAcceptProposal={vi.fn()}
        onRejectProposal={vi.fn()}
      />,
    );

    expect(screen.getByPlaceholderText('Reply to the clarification in your own words...')).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole('button').at(-1)!);

    expect(onResumeSend).toHaveBeenCalledTimes(1);
    expect(onSend).not.toHaveBeenCalled();
  });
});
