import { memo } from 'react';
import { ReactFlowProvider, Node } from '@xyflow/react';
import { WorkflowCanvas } from '@/components/workflows/canvas/WorkflowCanvas';
import { WorkflowLifecycleBar } from '@/components/workflows/lifecycle/WorkflowLifecycleBar';
import { Button } from '@/components/ui/button';
import { useUIStore } from '@/stores/uiStore';
import { WorkflowImportDialog } from '@/components/workflows/dialogs/WorkflowImportDialog';
import type { WorkflowVersion } from '@/types/workflow';
import type { WorkflowNodeData, WorkflowEdge } from '@/components/workflows/types';

interface ProposalPreview {
  proposal: {
    summary: string;
  };
}

interface WorkflowCanvasSectionProps {
  selectedWorkflowId: string | null;
  previewGraph: { nodes: Node<WorkflowNodeData>[]; edges: WorkflowEdge[] } | null;
  proposalPreview: ProposalPreview | null;
  lifecycleStatus: { isStale: boolean; isPublishable: boolean };
  workflowVersions: WorkflowVersion[];
  isLoadingWorkflowVersions: boolean;
  isExecuting: boolean;
  isPublishing: boolean;
  isRestoringVersion: boolean;
  canRun: boolean;
  canPublish: boolean;
  publishDisabledReason?: string;
  runDisabledReason?: string;
  isImportDialogOpen: boolean;
  onNodeClick?: (event: React.MouseEvent, node: Node<WorkflowNodeData>) => void;
  onNodeDrop?: (event: React.DragEvent) => void;
  onNodesRemoved?: () => void;
  onRunClick: () => void;
  onPublishClick: () => void;
  onVersionRestore: (versionId: number) => void;
  onImportWorkflow: (workflowData: unknown) => void;
  setImportDialogOpen: (open: boolean) => void;
  readOnly?: boolean;
  readOnlyReason?: string | null;
  onRetryLock?: () => void;
  versionRestoreDisabledReason?: string;
}

function ReadOnlyBanner({
  message,
  onRetryLock,
}: {
  message: string;
  onRetryLock?: () => void;
}) {
  return (
    <div className="absolute left-4 top-4 z-20 max-w-md rounded-xl border border-amber-300 bg-amber-50/95 px-4 py-3 shadow-lg">
      <p className="text-sm font-medium text-amber-950">Workflow is read-only</p>
      <p className="mt-1 text-xs text-amber-900">{message}</p>
      {onRetryLock && (
        <Button
          size="sm"
          variant="outline"
          className="mt-3 border-amber-300 bg-white/80 text-amber-950 hover:bg-white"
          onClick={onRetryLock}
        >
          Retry edit lock
        </Button>
      )}
    </div>
  );
}

export const WorkflowCanvasSection = memo(function WorkflowCanvasSection({
  selectedWorkflowId,
  previewGraph,
  proposalPreview,
  lifecycleStatus,
  workflowVersions,
  isLoadingWorkflowVersions,
  isExecuting,
  isPublishing,
  isRestoringVersion,
  canRun,
  canPublish,
  publishDisabledReason,
  runDisabledReason,
  isImportDialogOpen,
  onNodeClick,
  onNodeDrop,
  onNodesRemoved,
  onRunClick,
  onPublishClick,
  onVersionRestore,
  onImportWorkflow,
  setImportDialogOpen,
  readOnly = false,
  readOnlyReason = null,
  onRetryLock,
  versionRestoreDisabledReason,
}: WorkflowCanvasSectionProps) {
  const isPreviewActive = Boolean(previewGraph);
  const isCanvasReadOnly = isPreviewActive || readOnly;
  const rightPaneMode = useUIStore((state) => state.rightPaneMode);
  const isPaneOpen = rightPaneMode !== null;

  return (
    <ReactFlowProvider>
      <div className="flex-1 relative overflow-hidden">
        <WorkflowCanvas
          key={selectedWorkflowId ?? 'new'}
          previewGraph={previewGraph}
          onNodeClick={onNodeClick}
          onNodeDrop={isPreviewActive ? undefined : onNodeDrop}
          onNodesRemoved={isPreviewActive ? undefined : onNodesRemoved}
          readOnly={isCanvasReadOnly}
        />
        {!isPreviewActive && isCanvasReadOnly && (
          <ReadOnlyBanner
            message={readOnlyReason ?? 'Another member is editing this workflow.'}
            onRetryLock={onRetryLock}
          />
        )}
        <div
          className="absolute top-4 z-20 flex flex-col items-center gap-3 pointer-events-none -translate-x-1/2 transition-[left] duration-200 ease-in-out"
          style={{ left: `calc(50% - ${isPaneOpen ? 180 : 0}px)` }}
        >
          <div className="pointer-events-auto">
            <WorkflowLifecycleBar
              lifecycleStatus={lifecycleStatus}
              onRunClick={onRunClick}
              onPublishClick={onPublishClick}
              isExecuting={isExecuting}
              isPublishing={isPublishing}
              isRestoringVersion={isRestoringVersion}
              canRun={canRun}
              canPublish={canPublish}
              publishDisabledReason={publishDisabledReason}
              runDisabledReason={runDisabledReason}
              versionOptions={workflowVersions}
              isVersionsLoading={isLoadingWorkflowVersions}
              onVersionRestore={onVersionRestore}
              workflowId={selectedWorkflowId}
              versionRestoreDisabledReason={versionRestoreDisabledReason}
            />
          </div>
          {proposalPreview && (
            <div className="pointer-events-auto">
              <div className="bg-sky-900/90 text-white px-4 py-2 rounded-full shadow-lg max-w-xl text-center">
                <p className="text-sm font-medium">Previewing workflow proposal</p>
                <p className="text-xs text-slate-100 line-clamp-2">
                  {proposalPreview.proposal.summary}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Workflow Import Dialog */}
        <WorkflowImportDialog
          open={isImportDialogOpen}
          onOpenChange={setImportDialogOpen}
          onImport={onImportWorkflow}
        />
      </div>
    </ReactFlowProvider>
  );
});
