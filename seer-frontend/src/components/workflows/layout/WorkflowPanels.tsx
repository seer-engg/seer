import { memo } from 'react';
import type { ReactNode } from 'react';
import type { Node } from '@xyflow/react';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';
import { StandaloneChatPanel } from '@/components/workflows/panels/StandaloneChatPanel';

export interface PreviewGraphData {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
}
import { WorkflowCanvasSection } from '@/components/workflows/layout/WorkflowCanvasSection';
import { ResizablePanel } from '@/components/ui/resizable';

export const CanvasSection = memo(function CanvasSection({
  canvasProps,
  children,
}: {
  canvasProps: Record<string, unknown>;
  children?: ReactNode;
}) {
  return (
    <ResizablePanel defaultSize={72} minSize={40} className="relative flex flex-col">
      <WorkflowCanvasSection {...canvasProps} />
      {children}
    </ResizablePanel>
  );
});

export function LeftChatPanel({
  workflowId,
  workflowName,
  onWorkflowGraphSync,
  onRenameWorkflow,
  onDeleteWorkflow,
  readOnly = false,
}: {
  workflowId: string;
  workflowName: string;
  onWorkflowGraphSync: (graph: unknown) => void;
  onRenameWorkflow: (workflowId: string, newName: string) => Promise<void>;
  onDeleteWorkflow: (workflowId: string) => Promise<void>;
  readOnly?: boolean;
}) {
  return (
    <ResizablePanel defaultSize={28} minSize={15} maxSize={40} className="min-w-0">
      <StandaloneChatPanel
        workflowId={workflowId}
        workflowName={workflowName}
        onWorkflowGraphSync={onWorkflowGraphSync}
        onRenameWorkflow={onRenameWorkflow}
        onDeleteWorkflow={onDeleteWorkflow}
        readOnly={readOnly}
      />
    </ResizablePanel>
  );
}
