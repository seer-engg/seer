import { Node } from '@xyflow/react';
import { createContext, useContext } from 'react';
import { WorkflowEdge, WorkflowNodeData } from '../types';

export interface WorkflowCanvasContextValue {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
  updateNodeData: (nodeId: string, updates: Partial<WorkflowNodeData>) => void;
  readOnly: boolean;
}

export const WorkflowCanvasContext = createContext<WorkflowCanvasContextValue | null>(null);

export function useWorkflowCanvasContext(): WorkflowCanvasContextValue {
  const context = useContext(WorkflowCanvasContext);
  if (!context) {
    throw new Error('useWorkflowCanvasContext must be used within a WorkflowCanvasContext provider');
  }
  return context;
}


