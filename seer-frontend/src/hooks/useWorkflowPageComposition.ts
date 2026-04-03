import { Node } from '@xyflow/react';
import { WorkflowNodeData, WorkflowEdge, FunctionBlockSchema } from '@/components/workflows/types';
import { useWorkflowVersions } from '@/hooks/useWorkflowVersions';
import { useWorkflowTriggers } from '@/hooks/useWorkflowTriggers';
import { useWorkflowBlocks } from '@/hooks/useWorkflowBlocks';
import { useWorkflowInputs } from '@/hooks/useWorkflowInputs';
import { useGmailIntegration } from '@/hooks/useGmailIntegration';
import { useDiscordIntegration } from '@/hooks/useDiscordIntegration';
import { useSlackIntegration } from '@/hooks/useSlackIntegration';
import { useCalendarIntegration } from '@/hooks/useCalendarIntegration';
import { useBackendConnectionEffects } from '@/hooks/useBackendConnectionEffects';

export interface WorkflowHooksParams {
  selectedWorkflowId: string | null;
  functionBlockSchemas: FunctionBlockSchema[];
  functionBlocksMap: Map<string, FunctionBlockSchema>;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  setProposalPreview: (preview: unknown) => void;
}

export function useWorkflowHooks({
  selectedWorkflowId,
  functionBlockSchemas,
  functionBlocksMap,
  setNodes,
  setEdges,
  setProposalPreview,
}: WorkflowHooksParams) {
  const {
    versions: workflowVersions,
    isLoading: isLoadingWorkflowVersions,
  } = useWorkflowVersions(selectedWorkflowId);
  const {
    triggers: triggerCatalog,
    isLoadingTriggers,
    workflowTriggers,
    addTrigger,
    updateTrigger,
    removeTrigger,
  } = useWorkflowTriggers(selectedWorkflowId);
  const { availableBlocks, handleWorkflowGraphSync } = useWorkflowBlocks({
    functionBlockSchemas,
    functionBlocksMap,
    setNodes,
    setEdges,
    setProposalPreview,
    setLastRunVersionId: () => {},
  });

  return {
    workflowVersions,
    isLoadingWorkflowVersions,
    triggerCatalog,
    isLoadingTriggers,
    workflowTriggers,
    addTrigger,
    updateTrigger,
    removeTrigger,
    availableBlocks,
    handleWorkflowGraphSync,
  };
}

export function useWorkflowInitialization() {
  const { workflowInputsDef, inputFields } = useWorkflowInputs();
  const { gmailToolNames, gmailConnectionId } = useGmailIntegration();
  const { discordToolNames, discordConnectionId } = useDiscordIntegration();
  const { slackToolNames, slackConnectionId } = useSlackIntegration();
  const { calendarToolNames, calendarConnectionId } = useCalendarIntegration();
  useBackendConnectionEffects();

  return {
    workflowInputsDef,
    inputFields,
    gmailToolNames,
    gmailConnectionId,
    discordToolNames,
    discordConnectionId,
    slackToolNames,
    slackConnectionId,
    calendarToolNames,
    calendarConnectionId,
  };
}
