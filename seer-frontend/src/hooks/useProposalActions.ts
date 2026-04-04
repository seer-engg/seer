import { useQueryClient } from '@tanstack/react-query';
import { useCallback } from 'react';
import { toast } from '@/components/ui/sonner';
import { backendApiClient } from '@/lib/api-client';
import { chatKeys } from '@/lib/query-keys';
import { workflowSpecToGraph } from '@/lib/workflow-graph';
import { useChatStore, useTriggersStore } from '@/stores';
import type { WorkflowProposal, WorkflowProposalActionResponse } from '@/components/workflows/buildtypes';
import type { ChatSessionMessagesResult } from '@/hooks/useChatMessages';

export function useProposalActions(
  workflowId: string | null,
  onWorkflowGraphSync?: (graph: ReturnType<typeof workflowSpecToGraph>) => void,
) {
  const currentSessionId = useChatStore((state) => state.currentSessionId);
  const setProposalActionLoading = useChatStore((state) => state.setProposalActionLoading);
  const queryClient = useQueryClient();

  const updateProposalInMessages = useCallback(
    (updatedProposal: WorkflowProposal) => {
      useChatStore.setState((state) => ({
        transientMessages: state.transientMessages.map((message) =>
          message.proposal?.id === updatedProposal.id
            ? { ...message, proposal: updatedProposal }
            : message,
        ),
      }));

      if (!currentSessionId) {
        return;
      }

      queryClient.setQueryData<ChatSessionMessagesResult>(
        chatKeys.messagesBySession(currentSessionId),
        (prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            messages: prev.messages.map((message) =>
              message.proposal?.id === updatedProposal.id ? { ...message, proposal: updatedProposal } : message,
            ),
          };
        },
      );
    },
    [currentSessionId, queryClient],
  );

  const handleAcceptProposal = useCallback(
    async (proposalId: number) => {
      if (!workflowId) return;
      setProposalActionLoading(proposalId);
      try {
        const response = await backendApiClient.request<WorkflowProposalActionResponse>(
          `/api/nexus/${workflowId}/proposals/${proposalId}/accept`,
          { method: 'POST' },
        );
        updateProposalInMessages(response.proposal);
        // Load triggers from the accepted proposal spec into the triggers store
        if (workflowId && response.proposal?.spec?.triggers) {
          useTriggersStore.getState().loadTriggersFromSpec(workflowId, response.proposal.spec.triggers);
        }
        const specGraph = response.proposal?.spec ? workflowSpecToGraph(response.proposal.spec) : null;
        if (specGraph) {
          onWorkflowGraphSync?.(specGraph);
        } else if (response.workflow_graph) {
          onWorkflowGraphSync?.(response.workflow_graph);
        }
        toast.success('Proposal accepted');
        return response;
      } catch (error) {
        console.error('Failed to accept proposal:', error);
        toast.error('Failed to accept proposal');
      } finally {
        setProposalActionLoading(null);
      }
    },
    [workflowId, setProposalActionLoading, updateProposalInMessages, onWorkflowGraphSync],
  );

  const handleRejectProposal = useCallback(
    async (proposalId: number) => {
      if (!workflowId) return;
      setProposalActionLoading(proposalId);
      try {
        const response = await backendApiClient.request<WorkflowProposalActionResponse>(
          `/api/nexus/${workflowId}/proposals/${proposalId}/reject`,
          { method: 'POST' },
        );
        updateProposalInMessages(response.proposal);
        toast.success('Proposal rejected');
        return response;
      } catch (error) {
        console.error('Failed to reject proposal:', error);
        toast.error('Failed to reject proposal');
      } finally {
        setProposalActionLoading(null);
      }
    },
    [workflowId, setProposalActionLoading, updateProposalInMessages],
  );

  return {
    handleAcceptProposal,
    handleRejectProposal,
  };
}
