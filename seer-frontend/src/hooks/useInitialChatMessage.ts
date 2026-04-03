import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useChatStore, useUIStore } from '@/stores';
import type { WorkflowNavigationState } from '@/types/navigation';

/**
 * Hook to handle auto-sending initial chat messages when navigating from dashboard
 *
 * This enables the "dashboard → canvas" flow where:
 * 1. User enters a message in dashboard chat
 * 2. useDiscoveryChat creates a new workflow and navigates to canvas with message in state
 * 3. This hook picks up the message from navigation state
 * 4. Automatically sends it in the canvas chat panel
 *
 * Uses two effects to ensure proper timing:
 * - First effect: Processes navigation state and sets input
 * - Second effect: Triggers send after input is confirmed in state
 */
export function useInitialChatMessage(
  workflowId: string | null,
  handleSend: () => void
) {
  const location = useLocation();
  const hasProcessedNavigationRef = useRef(false);
  const shouldAutoSendRef = useRef(false);
  const setInput = useChatStore((state) => state.setInput);
  const setSelectedModel = useChatStore((state) => state.setSelectedModel);
  const setActiveRightPanelTab = useUIStore((state) => state.setActiveRightPanelTab);
  const input = useChatStore((state) => state.input);

  // Effect 1: Process navigation state and prepare chat input
  useEffect(() => {
    // Only run once per component mount to prevent duplicate sends
    if (hasProcessedNavigationRef.current) return;

    // Need valid workflow before processing
    if (!workflowId) return;

    const state = location.state as WorkflowNavigationState | null;

    if (state?.initialMessage) {
      // Mark as processed immediately to prevent re-runs
      hasProcessedNavigationRef.current = true;
      shouldAutoSendRef.current = true;

      // Set message in chat store
      setInput(state.initialMessage);
      if (state.initialModel) {
        setSelectedModel(state.initialModel);
      }

      // Switch to chat tab to show the conversation
      setActiveRightPanelTab('chat');

      // Clear navigation state to prevent re-send on browser refresh
      window.history.replaceState({}, document.title);
    }
  }, [workflowId, location.state, setInput, setSelectedModel, setActiveRightPanelTab]);

  // Effect 2: Trigger send after input is confirmed in state
  useEffect(() => {
    // Only trigger if we've flagged an auto-send and have valid input
    if (!shouldAutoSendRef.current || !input.trim() || !workflowId) {
      return;
    }

    // Reset flag immediately to prevent multiple sends
    shouldAutoSendRef.current = false;

    // Small delay to ensure component is fully mounted and handleSend has updated closure
    // This prevents timing issues where handleSend might have stale references
    setTimeout(() => {
      handleSend();
    }, 100);
  }, [input, workflowId, handleSend]);
}
