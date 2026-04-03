/**
 * ConfigPanel - Configuration panel for selected workflow node
 *
 * Displays node configuration UI in the right panel when a node is clicked.
 * Shows an empty state when no node is selected.
 */
import { Node } from '@xyflow/react';
import { Info, Settings } from 'lucide-react';

import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { BlockConfigPanel } from './BlockConfigPanel';
import { NodeIdEditor } from './NodeIdEditor';
import type { WorkflowNodeData, WorkflowNodeUpdateOptions, WorkflowEdge } from '../types';
import type { InputDef, TriggerSpec } from '@/types/workflow-spec';

function EmptyConfigState() {
  return (
    <div className="flex items-center justify-center flex-1">
      <div className="text-center space-y-3 p-6">
        <div className="w-12 h-12 mx-auto bg-muted rounded-full flex items-center justify-center">
          <Settings className="w-5 h-5 text-muted-foreground" />
        </div>
        <div>
          <p className="text-sm font-medium">No block selected</p>
          <p className="text-xs text-muted-foreground mt-1">
            Click a block on the canvas to configure it
          </p>
        </div>
      </div>
    </div>
  );
}

interface ConfigPanelProps {
  selectedNode: Node<WorkflowNodeData> | null;
  allNodes: Node<WorkflowNodeData>[];
  allEdges: WorkflowEdge[];
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
  onRenameNode?: (oldId: string, newId: string) => { success: boolean; error?: string };
  onClearSelection: () => void;
  workflowInputs?: Record<string, InputDef>;
  triggers?: TriggerSpec[];
  readOnly?: boolean;
}

export function ConfigPanel({
  selectedNode,
  allNodes,
  allEdges,
  onUpdate,
  onRenameNode,
  onClearSelection,
  workflowInputs,
  triggers,
  readOnly = false,
}: ConfigPanelProps) {
  if (!selectedNode) {
    return (
      <div className="flex flex-col h-full">
        <EmptyConfigState />
      </div>
    );
  }

  const existingNodeIds = allNodes.map((n) => n.id);

  return (
    <div className="flex flex-col h-full bg-background">
      {onRenameNode && (
        <div className="px-4 py-3 border-b border-border bg-muted/30 shrink-0">
          <div className="flex items-center gap-1 mb-1.5">
            <Label className="text-xs text-muted-foreground">Node ID</Label>
            <TooltipProvider delayDuration={200}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3 w-3 text-muted-foreground/70 cursor-help" />
                </TooltipTrigger>
                <TooltipContent side="right" className="max-w-[220px] text-xs">
                  <p className="font-medium mb-1">Naming rules:</p>
                  <ul className="list-disc pl-3.5 space-y-0.5 text-muted-foreground">
                    <li>Must be unique</li>
                    <li>Start with a letter</li>
                    <li>Letters, numbers, underscores only</li>
                    <li>Max 64 characters</li>
                  </ul>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          <NodeIdEditor
            nodeId={selectedNode.id}
            existingNodeIds={existingNodeIds}
            onRename={onRenameNode}
            disabled={readOnly}
          />
        </div>
      )}
      <ScrollArea className="flex-1">
        <div className="p-4">
          <BlockConfigPanel
            node={selectedNode}
            onUpdate={onUpdate}
            allNodes={allNodes}
            allEdges={allEdges}
            variant="inline"
            liveUpdate={!readOnly}
            liveUpdateDelayMs={350}
            workflowInputs={workflowInputs}
            triggers={triggers}
            autoSave={false}
            showSaveButton={false}
            readOnly={readOnly}
          />
        </div>
      </ScrollArea>
    </div>
  );
}
