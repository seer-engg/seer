import { useEffect, useState } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import { ReactFlowProvider } from '@xyflow/react';
import { WorkflowCanvas } from '@/components/workflows/canvas/WorkflowCanvas';
import { workflowSpecToGraph } from '@/lib/workflow-graph';
import { getBackendBaseUrl } from '@/lib/api-client';
import type { WorkflowGraphData } from '@/lib/workflow-graph';

interface PreviewNode {
  id: string;
  type: string;
  label: string;
  position: { x: number; y: number } | null;
}

interface PreviewEdge {
  source: string;
  target: string;
  type: string | null;
}

interface PreviewTrigger {
  id: string;
  key: string;
  mode: string;
  ui_meta: { position?: { x: number; y: number } } | null;
}

interface PreviewResponse {
  nodes: PreviewNode[];
  edges: PreviewEdge[];
  triggers: PreviewTrigger[];
  metadata: { name: string; description: string; icon: string | null };
}

/** Convert preview API response into a WorkflowSpec shape that workflowSpecToGraph understands. */
function previewToSpec(preview: PreviewResponse) {
  return {
    version: '2' as const,
    nodes: preview.nodes.map((n) => {
      const base = {
        id: n.id,
        type: n.type,
        ui: n.position ? { position: n.position } : undefined,
      };
      switch (n.type) {
        case 'tool':
          return { ...base, type: 'tool' as const, tool: n.label };
        case 'if':
          return { ...base, type: 'if' as const, condition: '' };
        case 'for_each':
          return { ...base, type: 'for_each' as const, items: '' };
        case 'hitl':
          return { ...base, type: 'hitl' as const, title: n.label };
        case 'browser':
          return { ...base, type: 'browser' as const, task: '' };
        case 'agent':
          return { ...base, type: 'agent' as const, inputs: {} };
        case 'mcp':
          return { ...base, type: 'mcp' as const, server: '', server_type: 'http' as const, tool: n.label, mcp_inputs: {} };
        case 'image_gen':
          return { ...base, type: 'image_gen' as const, inputs: {} };
        default:
          return { ...base, type: 'tool' as const, tool: n.label };
      }
    }),
    edges: preview.edges.map((e) => ({
      source: e.source,
      target: e.target,
      type: (e.type ?? 'default') as 'default',
    })),
    triggers: preview.triggers.map((t) => ({
      id: t.id,
      key: t.key,
      mode: t.mode,
      event_schema: {},
      meta: {},
      ui_meta: t.ui_meta ?? undefined,
    })),
  };
}

export default function EmbedWorkflow() {
  const { slug } = useParams<{ slug: string }>();
  const [searchParams] = useSearchParams();
  const [graph, setGraph] = useState<WorkflowGraphData | null>(null);
  const [meta, setMeta] = useState<PreviewResponse['metadata'] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const theme = searchParams.get('theme') ?? 'light';
  const bg = searchParams.get('bg');

  useEffect(() => {
    if (!slug) return;
    const baseUrl = getBackendBaseUrl();
    fetch(`${baseUrl}/api/public/templates/${slug}/preview`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json() as Promise<PreviewResponse>;
      })
      .then((data) => {
        setMeta(data.metadata);
        setGraph(workflowSpecToGraph(previewToSpec(data)));
      })
      .catch((e) => setError(e.message));
  }, [slug]);

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center text-muted-foreground text-sm">
        Template not found
      </div>
    );
  }

  if (!graph) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div
      className={`h-screen w-screen relative ${theme === 'dark' ? 'dark' : ''}`}
      style={bg === 'transparent' ? { background: 'transparent' } : undefined}
    >
      <ReactFlowProvider>
        <WorkflowCanvas previewGraph={graph} readOnly />
      </ReactFlowProvider>
      {meta && (
        <div className="absolute bottom-3 left-3 rounded-md bg-background/80 backdrop-blur-sm px-3 py-1.5 text-xs text-muted-foreground border">
          {meta.icon && <span className="mr-1.5">{meta.icon}</span>}
          {meta.name}
        </div>
      )}
    </div>
  );
}
