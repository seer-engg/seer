import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useChatStore } from '@/stores';
import { useWorkflowStore } from '@/stores/workflowStore';
import { generateWorkflowName } from '@/lib/workflow-names';

export function useDiscoveryChat() {
  const navigate = useNavigate();

  const input = useChatStore((state) => state.input);
  const isLoading = useChatStore((state) => state.isLoading);
  const selectedModel = useChatStore((state) => state.selectedModel);
  const setInput = useChatStore((state) => state.setInput);
  const setIsLoading = useChatStore((state) => state.setIsLoading);
  const createWorkflow = useWorkflowStore((state) => state.createWorkflow);

  const handleSend = useCallback(async (overrideInput?: string) => {
    const text = (overrideInput ?? input).trim();
    if (!text || isLoading) return;
    const messageContent = text;
    setInput('');
    setIsLoading(true);

    try {
      // Create blank workflow
      const workflow = await createWorkflow(generateWorkflowName(), {
        nodes: [],
        edges: [],
      });

      setIsLoading(false); // Reset loading state before navigation

      // Navigate with message state
      navigate(`/workflows/${workflow.workflow_id}`, {
        replace: true,
        state: {
          initialMessage: messageContent,
          initialModel: selectedModel,
        },
      });
    } catch (error) {
      console.error('Failed to create workflow:', error);
      setIsLoading(false);
    }
  }, [
    input,
    isLoading,
    selectedModel,
    setInput,
    setIsLoading,
    createWorkflow,
    navigate,
  ]);

  return { handleSend };
}
