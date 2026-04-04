/**
 * useAutoLayout Hook
 *
 * Provides a function to automatically organize the workflow graph
 * into a clean tree layout using the dagre algorithm.
 */
import { useCallback } from 'react';
import { useReactFlow } from '@xyflow/react';
import { useCanvasStore } from '@/stores';
import { getLayoutedElements } from '@/lib/graph-layout';

/**
 * Hook that provides auto-layout functionality for the workflow canvas.
 *
 * @returns Object containing:
 *   - organizeGraph: Function to trigger the layout reorganization
 *   - canOrganize: Boolean indicating if layout is possible (has nodes)
 */
export function useAutoLayout() {
  const nodes = useCanvasStore((state) => state.nodes);
  const edges = useCanvasStore((state) => state.edges);
  const setNodes = useCanvasStore((state) => state.setNodes);
  const markDirty = useCanvasStore((state) => state.markDirty);
  const { fitView } = useReactFlow();

  const organizeGraph = useCallback(() => {
    // Skip if no nodes to layout
    if (nodes.length === 0) return;

    // Compute new positions using dagre
    const { nodes: layoutedNodes } = getLayoutedElements(nodes, edges);

    // Update the canvas store with new node positions
    setNodes(layoutedNodes);

    // Mark as dirty so autosave picks up the changes
    markDirty();

    // Animate the view to show all nodes after layout
    // Small timeout ensures React Flow has processed the position updates
    setTimeout(() => {
      fitView({ duration: 500, padding: 0.2 });
    }, 50);
  }, [nodes, edges, setNodes, markDirty, fitView]);

  return {
    organizeGraph,
    canOrganize: nodes.length > 0,
  };
}
