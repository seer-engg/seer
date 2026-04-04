import { useCallback } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useChatStore } from '@/stores';

/**
 * Hook that provides a fixWithAI function to route errors to Nexus AI chat.
 *
 * Two paths:
 * - Same page (workflow canvas): sets pendingAutoSendMessage in chatStore
 * - Cross page (executions → workflow): navigates with initialMessage in location state
 */
export function useFixWithAI() {
  const navigate = useNavigate();
  const location = useLocation();
  const setInput = useChatStore((s) => s.setInput);
  const setPendingAutoSendMessage = useChatStore((s) => s.setPendingAutoSendMessage);

  const fixWithAI = useCallback(
    (error: string, workflowId?: string) => {
      const message = `Please fix my current workflow which is having the following error:\n\n${error}`;

      if (location.pathname.startsWith('/workflows/')) {
        // Same-page flow: set input + trigger auto-send (panel is always visible)
        setInput(message);
        setPendingAutoSendMessage(message);
      } else {
        // Cross-page flow: navigate to workflow with initialMessage in state
        if (!workflowId) return;
        navigate(`/workflows/${workflowId}`, { state: { initialMessage: message } });
      }
    },
    [location.pathname, navigate, setInput, setPendingAutoSendMessage],
  );

  return { fixWithAI };
}
