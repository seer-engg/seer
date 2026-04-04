import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

import { useChatStore } from '@/stores/chatStore';

import { ChatInput } from './ChatInput';

describe('ChatInput', () => {
  beforeEach(() => {
    useChatStore.getState().resetChatState();
    useChatStore.setState({
      input: 'Need help with Gmail routing',
    });
  });

  it('switches the composer into resume mode while clarification is pending', () => {
    render(
      <ChatInput
        onSend={vi.fn()}
        models={[]}
        isLoadingModels={false}
        isInterruptPending
      />,
    );

    expect(screen.getByPlaceholderText('Reply to the clarification in your own words...')).toBeEnabled();
    expect(screen.getAllByRole('button').at(-1)).toBeEnabled();
    expect(screen.getByText(/Resume mode: reply here or use the clarification card above/i)).toBeInTheDocument();
  });
});
