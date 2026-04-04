import { useEffect } from 'react';

import { ChatInput } from '@/components/workflows/chat/ChatInput';
import { SuggestedPrompts } from './SuggestedPrompts';
import { WorkflowGallery } from './WorkflowGallery';
import { useDiscoveryChat } from '@/hooks/useDiscoveryChat';
import { useAvailableModels } from '@/hooks/useAvailableModels';
import { useChatStore } from '@/stores';

export function DefaultChatView() {
  const { handleSend } = useDiscoveryChat();
  const { models, isLoadingModels } = useAvailableModels();

  const resetChatState = useChatStore((state) => state.resetChatState);

  useEffect(() => {
    // Reset chat state when entering dashboard to ensure clean slate
    resetChatState();
  }, [resetChatState]);

  const handleChipClick = (prompt: string) => {
    handleSend(prompt);
  };

  return (
    <div className="relative min-h-screen">
      {/* Ambient gradient - decorative background */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-[radial-gradient(ellipse_at_top,hsl(var(--seer)/0.08),transparent_70%)]" />
      </div>

      {/* Centered chat section - reduced to 45vh so gallery is visible without scrolling */}
      <div className="flex flex-col items-center justify-center min-h-[45vh] px-4 pt-10">
        <div className="w-full max-w-4xl">
          <div className="pt-16 pb-4">
            <h1 className="text-4xl font-bold mb-2">
              What's on your mind?
            </h1>
          </div>

          <SuggestedPrompts onSelectPrompt={handleChipClick} />
          <ChatInput
            onSend={handleSend}
            models={models}
            isLoadingModels={isLoadingModels}
          />
        </div>
      </div>

      {/* Workflow Gallery - positioned below the fold */}
      <div className="w-full max-w-4xl mx-auto px-4 pb-8">
        <WorkflowGallery />
      </div>
    </div>
  );
}
