import { BackendAPIError } from '@/lib/api-client';
import { workflowSpecToGraph } from '@/lib/workflow-graph';
import { toast } from '@/components/ui/sonner';
import type { ChatMessage, WorkflowProposal } from '../buildtypes';
import type { JsonObject } from '@/types/workflow-spec';
import type { Node, Edge } from '@xyflow/react';
import { getDisplayableAssistantMessage } from '../utils';

export function updateSessionIds(params: {
  sessionId?: number;
  threadId?: string;
  currentSessionId: number | null;
  currentThreadId: string | null;
  setCurrentSessionId: (id: number) => void;
  setCurrentThreadId: (id: string) => void;
}) {
  if (params.sessionId && params.sessionId !== params.currentSessionId) {
    params.setCurrentSessionId(params.sessionId);
  }
  if (params.threadId && params.threadId !== params.currentThreadId) {
    params.setCurrentThreadId(params.threadId);
  }
}

export function handleProposalResponse(
  proposal: WorkflowProposal | null | undefined,
  proposalError: string | null | undefined,
  setProposalPreview: (preview: { proposal: WorkflowProposal; graph: ReturnType<typeof workflowSpecToGraph> } | null) => void,
) {
  if (!proposal || !proposal.spec || proposalError) {
    setProposalPreview(null);
    return;
  }

  try {
    const previewGraph = workflowSpecToGraph(proposal.spec);
    setProposalPreview({ proposal, graph: previewGraph });
  } catch (graphError) {
    console.error('Failed to build workflow preview from proposal:', graphError);
    toast.error('Failed to preview workflow proposal');
    setProposalPreview(null);
  }
}

/**
 * Finds the most recent pending proposal from a list of messages.
 * Returns null if no pending proposal exists.
 */
export function findPendingProposalFromMessages(
  messages: ChatMessage[]
): WorkflowProposal | null {
  // Iterate in reverse to find the most recent pending proposal
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.proposal && msg.proposal.status === 'pending') {
      return msg.proposal;
    }
  }
  return null;
}

/**
 * Restores proposal preview from a pending proposal.
 * Returns true if preview was restored, false otherwise.
 */
export function restoreProposalPreview(
  proposal: WorkflowProposal | null,
  setProposalPreview: (preview: { proposal: WorkflowProposal; graph: ReturnType<typeof workflowSpecToGraph> } | null) => void,
): boolean {
  if (!proposal || !proposal.spec) {
    return false;
  }

  try {
    const previewGraph = workflowSpecToGraph(proposal.spec);
    setProposalPreview({ proposal, graph: previewGraph });
    return true;
  } catch (graphError) {
    console.error('Failed to restore workflow preview from proposal:', graphError);
    return false;
  }
}

export function createAssistantMessage(params: {
  response: string;
  proposal?: WorkflowProposal | null;
  proposalError?: string | null;
  thinking?: string[];
  interruptRequired?: boolean;
  interruptData?: JsonObject;
  selectedModel: string | null;
}): ChatMessage {
  const displayContent = getDisplayableAssistantMessage(params.response, params.proposal?.summary);
  return {
    role: 'assistant',
    content: displayContent,
    proposal: params.proposal || undefined,
    proposalError: params.proposalError || undefined,
    thinking: params.thinking,
    interruptRequired: params.interruptRequired,
    interruptData: params.interruptData,
    timestamp: new Date(),
    model: params.selectedModel,
  };
}

export function isTimeoutError(error: unknown): boolean {
  return (
    (error instanceof Error && error.name === 'AbortError') ||
    (error instanceof BackendAPIError && error.status === 504)
  );
}

/**
 * Type guard to check if error is a BackendAPIError
 */
function isBackendAPIError(error: unknown): error is BackendAPIError {
  return error instanceof BackendAPIError;
}

/**
 * Type guard to check if error response is a cost cap error wrapped in RFC 7807 Problem Details
 */
function isCostCapErrorResponse(response: unknown): boolean {
  return (
    typeof response === 'object' &&
    response !== null &&
    'detail' in response &&
    typeof response.detail === 'object' &&
    response.detail !== null &&
    'detail' in response.detail &&
    typeof response.detail.detail === 'object' &&
    response.detail.detail !== null &&
    'error' in response.detail.detail &&
    response.detail.detail.error === 'run_cost_cap_exceeded'
  );
}

/**
 * Extract cost cap details from RFC 7807 Problem Details error response
 */
function extractCostCapDetails(response: unknown): {
  accumulated_cost: number;
  cost_cap: number;
  run_type: 'chat' | 'workflow';
} | null {
  if (!isCostCapErrorResponse(response)) return null;

  const errorResponse = response as {
    detail: {
      detail: {
        accumulated_cost?: number;
        cost_cap?: number;
        run_type?: string;
      };
    };
  };

  return {
    accumulated_cost: errorResponse.detail.detail.accumulated_cost ?? 0,
    cost_cap: errorResponse.detail.detail.cost_cap ?? 0,
    run_type: (errorResponse.detail.detail.run_type as 'chat' | 'workflow') ?? 'chat',
  };
}

export function createErrorMessage(error: unknown): ChatMessage {
  // Check for timeout error
  if (isTimeoutError(error)) {
    return {
      role: 'assistant',
      content: 'Request timed out. Please try again with a shorter message.',
      timestamp: new Date(),
      error: {
        type: 'timeout',
      },
    };
  }

  // Check for cost cap exceeded error (402)
  if (isBackendAPIError(error) && error.status === 402) {
    const details = extractCostCapDetails(error.response);

    if (details) {
      return {
        role: 'assistant',
        content: 'This chat exceeded your cost limit.',
        timestamp: new Date(),
        error: {
          type: 'cost_cap_exceeded',
          details,
        },
      };
    }
  }

  // Generic fallback
  return {
    role: 'assistant',
    content: 'Sorry, I encountered an error. Please try again.',
    timestamp: new Date(),
    error: {
      type: 'generic',
    },
  };
}

export function prepareWorkflowState(nodes: Node[], edges: Edge[]) {
  return {
    nodes: nodes.map((n) => ({ id: n.id, type: n.type, data: n.data, position: n.position })),
    edges: edges.map((e) => {
      const branch = e.data?.branch;
      return { id: e.id, source: e.source, target: e.target, ...(branch ? { data: { branch } } : {}) };
    }),
  };
}
