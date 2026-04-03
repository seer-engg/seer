/**
 * ContextualPane - Handles both node config (full-screen modal) and executions (slide-in panel).
 *
 * - Node Config: Opens as a full-screen modal for ample configuration space
 * - Executions: Opens as a slide-in panel from the right edge
 */
import { useEffect } from 'react';
import { X, RefreshCw } from 'lucide-react';
import { useQueryClient, useIsFetching } from '@tanstack/react-query';
import { workflowRunKeys } from '@/lib/query-keys';
import type { Node } from '@xyflow/react';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useUIStore } from '@/stores/uiStore';
import { useCanvasStore, useTriggersStore } from '@/stores';
import { ConfigPanel } from './ConfigPanel';
import { ExecutionsPanel } from '../executions/ExecutionsPanel';
import type { WorkflowNodeData, WorkflowEdge } from '../types';
import type { TriggerSpec } from '@/types/workflow-spec';

interface ExecutionsPaneHeaderProps {
  onRefresh: () => void;
  onClose: () => void;
  isRefreshing: boolean;
}

function ExecutionsPaneHeader({ onRefresh, onClose, isRefreshing }: ExecutionsPaneHeaderProps) {
  return (
    <div className="h-14 border-b border-border flex items-center justify-between px-4 bg-card shrink-0">
      <h3 className="text-sm font-medium">Executions</h3>
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={onRefresh}
          className="h-8 w-8 shrink-0"
          aria-label="Refresh executions"
          disabled={isRefreshing}
        >
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          onClick={onClose}
          className="h-8 w-8 shrink-0"
          aria-label="Close pane"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

interface ContextualPaneProps {
  selectedNode: Node<WorkflowNodeData> | null;
  allNodes: Node<WorkflowNodeData>[];
  allEdges: WorkflowEdge[];
  onUpdate: (nodeId: string, updates: Partial<WorkflowNodeData>) => void;
  workflowId: string | null;
  workflowInputsDef: unknown;
  triggers: TriggerSpec[];
  readOnly?: boolean;
}

export function ContextualPane({
  selectedNode,
  allNodes,
  allEdges,
  onUpdate,
  workflowId,
  workflowInputsDef,
  triggers,
  readOnly = false,
}: ContextualPaneProps) {
  const rightPaneMode = useUIStore((state) => state.rightPaneMode);
  const closeRightPane = useUIStore((state) => state.closeRightPane);
  const setSelectedNodeId = useCanvasStore((state) => state.setSelectedNodeId);
  const renameNode = useCanvasStore((state) => state.renameNode);
  const renameTrigger = useTriggersStore((state) => state.renameTrigger);

  const isConfigOpen = rightPaneMode === 'config';
  const isExecutionsOpen = rightPaneMode === 'executions';

  // Close executions panel on Escape key
  useEffect(() => {
    if (!isExecutionsOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeRightPane();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isExecutionsOpen, closeRightPane]);

  const handleConfigOpenChange = (open: boolean) => {
    if (!open) {
      closeRightPane();
      setSelectedNodeId(null);
    }
  };

  const queryClient = useQueryClient();
  const isRefreshing = useIsFetching({ queryKey: workflowRunKeys.list(workflowId) }) > 0;

  const handleCloseExecutions = () => {
    closeRightPane();
  };

  const handleRefreshExecutions = () => {
    queryClient.invalidateQueries({ queryKey: workflowRunKeys.list(workflowId) });
  };

  const handleRenameNode = (oldId: string, newId: string) => {
    const targetNode = allNodes.find((n) => n.id === oldId);
    if (targetNode?.data?.type === 'trigger' && workflowId) {
      renameTrigger(workflowId, oldId, newId);
    }
    return renameNode(oldId, newId);
  };

  const handleClearSelection = () => {
    closeRightPane();
    setSelectedNodeId(null);
  };

  return (
    <>
      {/* Node Config - Full-screen modal */}
      <Dialog open={isConfigOpen} onOpenChange={handleConfigOpenChange}>
        <DialogContent className="w-[90vw] max-w-[1200px] h-[90vh] max-h-[800px] flex flex-col p-0 gap-0">
          <DialogHeader className="h-14 border-b border-border flex flex-row items-center justify-between px-6 bg-card shrink-0">
            <DialogTitle className="text-sm font-medium">Node Config</DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-hidden">
            <ConfigPanel
              selectedNode={selectedNode}
              allNodes={allNodes}
              allEdges={allEdges}
              onUpdate={onUpdate}
              onRenameNode={handleRenameNode}
              onClearSelection={handleClearSelection}
              workflowInputs={workflowInputsDef as Record<string, import('@/types/workflow-spec').InputDef>}
              triggers={triggers}
              readOnly={readOnly}
            />
          </div>
        </DialogContent>
      </Dialog>

      {/* Executions - Slide-in panel */}
      <div
        className={`absolute right-0 top-0 h-full w-[360px] z-30 bg-card border-l border-border shadow-2xl flex flex-col transition-transform duration-200 ease-in-out ${
          isExecutionsOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        aria-hidden={!isExecutionsOpen}
      >
        <ExecutionsPaneHeader
          onRefresh={handleRefreshExecutions}
          onClose={handleCloseExecutions}
          isRefreshing={isRefreshing}
        />
        <div className="flex-1 overflow-hidden">
          <ExecutionsPanel workflowId={workflowId} isOpen={isExecutionsOpen} />
        </div>
      </div>
    </>
  );
}
