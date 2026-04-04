---
name: react-flow-builder
description: React Flow best practices for workflow canvas in seer-frontend. Use when modifying WorkflowCanvas, adding new block types, debugging node rendering issues, or optimizing React Flow performance.
allowed-tools: Read, Grep, Glob
---

# React Flow Builder Skill

This Skill helps you work with React Flow (@xyflow/react) for building the workflow canvas in seer-frontend. Use this when creating new block nodes, modifying canvas behavior, or debugging rendering issues.

## Core React Flow Architecture

### 1. **WorkflowCanvas Component** (`src/components/workflows/canvas/WorkflowCanvas.tsx`)

The main canvas component that orchestrates React Flow.

**Key patterns:**
- Uses Zustand `canvasStore` for state management
- Supports both editable and read-only modes
- Integrates custom node types (tool, llm, if_else, for_loop, trigger)
- Handles node/edge changes with `applyNodeChanges` and `applyEdgeChanges`

**Node type registration:**
```tsx
const nodeTypes = {
  tool: ToolBlockNode,
  llm: LLMBlockNode,
  if_else: IfElseBlockNode,
  for_loop: ForLoopBlockNode,
  trigger: TriggerBlockNode,
};
```

**State management pattern:**
```tsx
const {
  nodes,
  edges,
  setNodes,
  setEdges,
  selectedNodeId,
  updateNode,
} = useCanvasStore(
  useShallow((state) => ({
    nodes: state.nodes,
    edges: state.edges,
    setNodes: state.setNodes,
    setEdges: state.setEdges,
    selectedNodeId: state.selectedNodeId,
    updateNode: state.updateNode,
  })),
);
```

**Props interface:**
- `triggerNodes`: Optional trigger nodes
- `previewGraph`: Optional preview graph for read-only display
- `onNodeDoubleClick`: Handler for node double-clicks (opens config dialog)
- `onNodeDrop`: Handler for drag-and-drop block placement
- `className`: Optional CSS classes
- `readOnly`: Disables editing when true

### 2. **BaseBlockNode Component** (`src/components/workflows/blocks/BaseBlockNode.tsx`)

The foundation for all custom block nodes.

**Key patterns:**
- Wrapped in `memo()` for performance optimization
- Uses React Flow `Handle` components for connections
- Supports customizable colors, icons, and handle positions
- Selected state triggers visual feedback (border, shadow, ring)

**Props interface:**
```tsx
interface BaseBlockNodeProps extends NodeProps<WorkflowNode> {
  icon?: React.ReactNode;           // Icon for the block
  color?: string;                    // Color theme (primary, success, warning, etc.)
  handles?: {
    inputs?: string[];               // Input handle IDs
    outputs?: string[];              // Output handle IDs
  };
  children?: React.ReactNode;        // Block content (badges, config, etc.)
  minWidth?: string;                 // Minimum width (default: 320px)
}
```

**Visual feedback pattern:**
```tsx
className={cn(
  'relative px-4 py-3 rounded-lg border-2 transition-[border,shadow,ring]',
  selected
    ? 'border-primary shadow-lg ring-2 ring-primary ring-offset-2'
    : 'border-border bg-card hover:border-primary/50',
)}
```

**Handle positioning:**
- **Input handle**: Left side, absolute position `left: -8px`, centered vertically
- **Output handle**: Right side, absolute position `right: -8px`, centered vertically
- Both handles: `!w-3 !h-3 !bg-border !border-2 !border-background`

### 3. **Zustand Canvas Store** (`src/stores/canvasStore.ts`)

Manages canvas state including nodes, edges, and selection.

**Key state:**
```tsx
interface CanvasStore {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
  selectedNodeId: string | null;
  setNodes: (nodes: Node<WorkflowNodeData>[]) => void;
  setEdges: (edges: WorkflowEdge[]) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
  updateNode: (nodeId: string, updates: Partial<WorkflowNodeData>) => void;
}
```

**State synchronization:**
- Nodes and edges are stored in Zustand store
- React Flow receives nodes/edges as props
- Changes are applied through store actions
- `useShallow` from zustand/shallow prevents unnecessary re-renders

### 4. **Workflow Store Integration** (`src/stores/workflowStore.ts`)

Manages workflow metadata and API interactions.

**Key patterns:**
- Uses `graphToWorkflowSpec` and `workflowSpecToGraph` for serialization
- Handles workflow CRUD operations
- Manages versions and draft states
- Executes workflows via backend API

**Graph conversion:**
```tsx
// Convert React Flow graph to backend spec
const spec = graphToWorkflowSpec(graph);

// Convert backend spec to React Flow graph
const graph = workflowSpecToGraph(spec);
```

## Creating a New Block Type

Follow these steps to add a new block type to the workflow canvas:

### Step 1: Define the Block Component

Create a new file in `src/components/workflows/blocks/` (e.g., `MyBlockNode.tsx`):

```tsx
import { memo } from 'react';
import { NodeProps, type Node } from '@xyflow/react';
import { MyIcon } from 'lucide-react';
import { BaseBlockNode } from './BaseBlockNode';
import { WorkflowNodeData } from '../types';
import { Badge } from '@/components/ui/badge';

type MyBlockNode = Node<WorkflowNodeData>;

export const MyBlockNode = memo(function MyBlockNode(props: NodeProps<MyBlockNode>) {
  const { data } = props;

  return (
    <BaseBlockNode
      {...props}
      icon={<MyIcon className="w-4 h-4 text-primary" />}
      color="primary"
      handles={{
        inputs: ['input'],
        outputs: ['output'],
      }}
    >
      {/* Optional: Add badges or status indicators */}
      <Badge variant="secondary" className="text-[10px]">
        {data.config?.someProperty}
      </Badge>
    </BaseBlockNode>
  );
});
```

### Step 2: Register the Node Type

Add the new node type to `canvas/WorkflowCanvas.tsx`:

```tsx
import { MyBlockNode } from './blocks/MyBlockNode';

const nodeTypes = {
  tool: ToolBlockNode,
  llm: LLMBlockNode,
  if_else: IfElseBlockNode,
  for_loop: ForLoopBlockNode,
  trigger: TriggerBlockNode,
  my_block: MyBlockNode, // Add your new block here
};
```

### Step 3: Define Block Data Type

Update `src/components/workflows/types.ts` with the new block config:

```tsx
export interface MyBlockConfig {
  someProperty: string;
  anotherProperty?: number;
}

// Add to WorkflowNodeData type union
export type WorkflowNodeData = {
  type: 'my_block';
  label: string;
  config: MyBlockConfig;
  // ... other common fields
} | /* other block types */;
```

### Step 4: Add Block to Palette

Update the block palette to include the new block for drag-and-drop:

```tsx
const blockPalette = [
  // ... existing blocks
  {
    type: 'my_block',
    label: 'My Block',
    icon: <MyIcon className="w-4 h-4" />,
    defaultConfig: {
      someProperty: 'default',
    },
  },
];
```

## Common React Flow Patterns

### Performance Optimization

**Use memo() for custom nodes:**
```tsx
export const MyBlockNode = memo(function MyBlockNode(props) {
  // Component implementation
});
```

**Use useShallow for store subscriptions:**
```tsx
const { nodes, edges } = useCanvasStore(
  useShallow((state) => ({
    nodes: state.nodes,
    edges: state.edges,
  })),
);
```

**Prevent event bubbling in interactive children:**
```tsx
<div onPointerDown={(event) => event.stopPropagation()}>
  <Button>Click Me</Button>
</div>
```

### Connection Validation

Validate connections before allowing edges to be created:

```tsx
const isValidConnection = useCallback((connection: Connection) => {
  // Prevent self-connections
  if (connection.source === connection.target) {
    return false;
  }

  // Custom validation logic
  const sourceNode = nodes.find((n) => n.id === connection.source);
  const targetNode = nodes.find((n) => n.id === connection.target);

  // Add your validation rules here
  return true;
}, [nodes]);

<ReactFlow
  isValidConnection={isValidConnection}
  // ... other props
/>
```

### Node Selection Handling

Highlight selected nodes in the canvas:

```tsx
const renderedNodes = useMemo(() => {
  return nodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      selected: node.id === selectedNodeId,
    },
  }));
}, [nodes, selectedNodeId]);
```

### Edge Styling

Customize edge appearance with markers and styles:

```tsx
const edges = useMemo(() => {
  return workflowEdges.map((edge) => ({
    ...edge,
    type: 'smoothstep',
    animated: false,
    markerEnd: {
      type: MarkerType.ArrowClosed,
      width: 20,
      height: 20,
    },
  }));
}, [workflowEdges]);
```

### Drag-and-Drop from Palette

Implement drop handler for adding blocks:

```tsx
const onDrop = useCallback(
  (event: React.DragEvent) => {
    event.preventDefault();
    const blockData = JSON.parse(event.dataTransfer.getData('application/json'));
    const position = screenToFlowPosition({
      x: event.clientX,
      y: event.clientY,
    });

    onNodeDrop?.(blockData, position);
  },
  [onNodeDrop, screenToFlowPosition],
);

<div onDrop={onDrop} onDragOver={(e) => e.preventDefault()}>
  <ReactFlow />
</div>
```

## Block Node Design Guidelines

### Visual Hierarchy

1. **Block container**: 2px border, rounded corners, padding
2. **Icon area**: 32×32px, rounded, semi-transparent background
3. **Label**: Font medium, text-sm, truncate overflow
4. **Badges**: Minimal design, text-[10px], subtle colors (see .claude/rules/component-design.md)

### Color Coding by Block Type

Use semantic colors for different block types:
- **LLM blocks**: `seer` (purple) - AI-powered operations
- **Tool blocks**: `success` (emerald) - External integrations
- **Logic blocks** (if/else, for loop): `primary` (default) - Control flow
- **Trigger blocks**: `warning` (amber) - Event triggers

### Handle Placement Rules

- **Single input/output**: Centered on left/right edges
- **Multiple inputs**: Vertically distributed on left edge
- **Multiple outputs**: Vertically distributed on right edge
- **Conditional outputs**: Labeled handles (e.g., "then", "else")

## Debugging React Flow Issues

### Common Issues and Solutions

**1. Nodes not rendering:**
- Verify node type is registered in `nodeTypes` object
- Check that node data matches expected `WorkflowNodeData` type
- Ensure node has valid `id`, `type`, `position`, and `data` fields

**2. Selection not updating:**
- Check that `selectedNodeId` is synchronized with store
- Verify `selected` prop is passed to custom node components
- Ensure selection styles are applied conditionally

**3. Edges not connecting:**
- Verify handles have correct `type` prop (target/source)
- Check handle positioning (absolute positioning required)
- Implement `isValidConnection` for custom validation

**4. Performance issues with large graphs:**
- Use `memo()` for all custom node components
- Use `useShallow` for store subscriptions
- Avoid expensive calculations in render functions
- Consider virtualization for very large graphs (100+ nodes)

**5. State not persisting:**
- Verify store actions are being called correctly
- Check that `updateNode` updates both store and React Flow state
- Ensure `applyNodeChanges` and `applyEdgeChanges` are used for React Flow changes

## Testing Workflow Canvas

### Manual Testing Checklist

- [ ] Nodes render correctly with proper icons and labels
- [ ] Selection highlights the correct node
- [ ] Edges connect properly between handles
- [ ] Drag-and-drop adds blocks at correct positions
- [ ] Node configuration dialog opens on double-click
- [ ] Canvas pans and zooms smoothly
- [ ] Read-only mode disables editing
- [ ] Dark mode renders correctly

### Integration Testing

Test workflow canvas with backend:
```bash
# Start frontend dev server
npm run dev

# Start backend API (in seer directory)
cd /Users/pika/Projects/seer
python -m api.main
```

## Key Files Reference

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/components/workflows/canvas/WorkflowCanvas.tsx` | Main canvas component | Adding global canvas features |
| `src/components/workflows/blocks/BaseBlockNode.tsx` | Base block component | Changing common block behavior |
| `src/components/workflows/blocks/*.tsx` | Custom block nodes | Adding/modifying specific block types |
| `src/stores/canvasStore.ts` | Canvas state management | Adding canvas state or actions |
| `src/stores/workflowStore.ts` | Workflow CRUD operations | Adding workflow API calls |
| `src/components/workflows/types.ts` | Type definitions | Adding new block types or configs |
| `.claude/rules/component-design.md` | UI component patterns | Following badge/button patterns |
| `.claude/rules/color-theming.md` | Color system | Applying semantic colors |

## Quick Checklist

When working with React Flow:
- [ ] Custom nodes are wrapped in `memo()`
- [ ] Node types are registered in `nodeTypes` object
- [ ] Handles have correct positioning and styling
- [ ] Store actions are used for state updates
- [ ] Selection state is synchronized
- [ ] Event bubbling is prevented for interactive children
- [ ] Color coding follows semantic color system
- [ ] Dark mode is supported
- [ ] Performance is optimized for large graphs
