/* eslint-disable max-lines */
import type { Node } from '@xyflow/react';

import type { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';
import type { JsonObject, JsonValue, TriggerSpec, WorkflowNode, WorkflowSpec, WorkflowSpecEdge, EdgeType } from '@/types/workflow-spec';
import { getUserDefaultModel } from '@/lib/default-model';

export interface WorkflowGraphData {
  nodes: Node<WorkflowNodeData>[];
  edges: WorkflowEdge[];
}

type FlowNode = Node<WorkflowNodeData>;
type BranchLabel = 'true' | 'false' | 'loop' | 'exit';

interface GraphContext {
  nodeById: Map<string, FlowNode>;
  outgoing: Map<string, WorkflowEdge[]>;
  incoming: Map<string, WorkflowEdge[]>;
}


/**
 * Safely serializes the current canvas graph into a WorkflowSpec payload.
 * UI metadata (positions, handles) is stored in entity-level meta properties.
 */
export function graphToWorkflowSpec(
  graph: WorkflowGraphData,
  existing?: WorkflowSpec,
  triggers?: TriggerSpec[],
): WorkflowSpec {
  const nodes = buildWorkflowNodes(graph);

  // Convert edges to backend format, including trigger edges
  const edges = graph.edges.map(edge => {
    const sourceNode = graph.nodes.find(n => n.id === edge.source);
    const isTriggerEdge = sourceNode?.data?.type === 'trigger';
    return convertEdgeToSpecEdge(edge, isTriggerEdge);
  });

  // Update triggers with ui_meta containing position
  const triggersWithUiMeta = (triggers ?? existing?.triggers ?? []).map(trigger => {
    const triggerNode = graph.nodes.find(n => n.id === trigger.id && n.data?.type === 'trigger');
    return {
      ...trigger,
      ui_meta: triggerNode ? { ...trigger.ui_meta, position: triggerNode.position } : (trigger.ui_meta ?? {}),
    };
  });

  return {
    version: existing?.version ?? '2',  // Use V2 for edge support
    nodes,
    edges,
    triggers: triggersWithUiMeta,
  };
}

/**
 * Reconstructs ReactFlow nodes/edges from a WorkflowSpec payload.
 * Reads UI metadata (positions, handles) from entity-level meta properties.
 */
export function workflowSpecToGraph(spec: WorkflowSpec): WorkflowGraphData {
  return stripInputNodes(buildGraphFromSpec(spec));
}

function buildWorkflowNodes(graph: WorkflowGraphData): WorkflowNode[] {
  // Filter out trigger nodes - they represent workflow entry points (webhooks, schedules, etc.)
  // and are stored separately in the workflow metadata, not as executable workflow nodes.
  const executableNodes = graph.nodes.filter((node) => node.data?.type !== 'trigger');

  // With edge-based architecture, we simply convert all nodes individually.
  // The edges define the execution order and control flow, not the node array order.
  return executableNodes.map((node) => convertNodeFlat(node));
}

/**
 * Converts a single ReactFlow node to a WorkflowNode without traversing edges.
 * Used in the edge-based architecture where control flow is defined by edges.
 */
function convertNodeFlat(node: FlowNode): WorkflowNode {
  switch (node.data?.type) {
    case 'tool':
      return createToolNode(node);
    case 'if_else':
      return createIfNodeFlat(node);
    case 'for_loop':
      return createForEachNodeFlat(node);
    case 'mcp':
      return createMcpNode(node);
    case 'hitl':
      return createHitlNode(node);
    case 'browser':
      return createBrowserNode(node);
    case 'image_gen':
      return createImageGenNode(node);
    case 'agent':
      return createAgentNode(node);
    case 'ea_bot':
      return createEABotNode(node);
    default:
      throw new Error(`Unsupported block type '${node.data?.type ?? 'unknown'}' on node '${node.id}'.`);
  }
}

function convertNode(node: FlowNode): WorkflowNode {
  switch (node.data?.type) {
    case 'tool':
      return createToolNode(node);
    case 'if_else':
      return createIfNodeFlat(node);
    case 'for_loop':
      return createForEachNodeFlat(node);
    case 'mcp':
      return createMcpNode(node);
    case 'hitl':
      return createHitlNode(node);
    case 'browser':
      return createBrowserNode(node);
    case 'image_gen':
      return createImageGenNode(node);
    case 'agent':
      return createAgentNode(node);
    case 'ea_bot':
      return createEABotNode(node);
    default:
      throw new Error(`Unsupported block type '${node.data?.type ?? 'unknown'}' on node '${node.id}'.`);
  }
}

function createToolNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const toolName = config.tool_name || config.toolName;
  if (!toolName) {
    throw new Error(`Tool block '${node.data?.label ?? node.id}' is missing a tool selection.`);
  }

  const params = isRecord(config.params)
    ? (config.params as Record<string, JsonValue>)
    : {};

  const inlineSchema = buildInlineSchemaSpec(config.output_schema);
  const expectOutput = inlineSchema
    ? {
        mode: 'json' as const,
        schema: inlineSchema,
      }
    : undefined;

  return {
    id: node.id,
    type: 'tool',
    tool: toolName,
    inputs: params,
    expect_outputs: expectOutput,
    connection_id: config.connection_id ?? null,  // Multi-account OAuth support
    ui: {
      position: node.position,
    },
  };
}

function serializeMcpStdio(config: Record<string, unknown>): { server: string; auth: Record<string, JsonValue> | undefined } {
  const parts = [config.stdio_command as string, ...((config.stdio_args as string[]) || [])];
  const server = parts.filter(Boolean).join(' ');
  const envVars = config.stdio_env as Record<string, string> | undefined;
  const hasEnv = envVars && Object.keys(envVars).length > 0;
  return { server, auth: hasEnv ? { env: envVars as unknown as JsonValue } as Record<string, JsonValue> : undefined };
}

function serializeMcpHttp(config: Record<string, unknown>): { server: string; auth: Record<string, JsonValue> | undefined } {
  const server = String(config.server || '');
  const headers = config.http_headers as Record<string, string> | undefined;
  const hasHeaders = headers && Object.keys(headers).length > 0;
  if (hasHeaders) {
    return { server, auth: { headers: headers as unknown as JsonValue } as Record<string, JsonValue> };
  }
  const fallbackAuth = isRecord(config.auth) ? (config.auth as Record<string, JsonValue>) : undefined;
  return { server, auth: fallbackAuth };
}

function createMcpNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const serverType = (config.server_type as 'http' | 'stdio') || 'http';

  const { server, auth } = serverType === 'stdio' && config.stdio_command
    ? serializeMcpStdio(config)
    : serializeMcpHttp(config);

  return {
    id: node.id,
    type: 'mcp',
    server,
    server_type: serverType,
    tool: String(config.tool || ''),
    inputs: isRecord(config.inputs)
      ? (config.inputs as Record<string, JsonValue>)
      : {},
    auth,
    ...(config.mcp_server_config_id ? { mcp_server_config_id: config.mcp_server_config_id as number } : {}),
    ui: {
      position: node.position,
    },
  };
}

type HitlInputType = 'single_choice' | 'multi_choice' | 'text' | 'number' | 'boolean';

function createHitlNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};

  // Convert delivery channels
  const deliveryChannels = Array.isArray(config.delivery_channels)
    ? config.delivery_channels.map((channel: { type: string; gmail?: { recipient_email: string } }) => ({
        type: channel.type as 'platform' | 'gmail',
        gmail: channel.gmail
          ? { recipient_email: String(channel.gmail.recipient_email || '') }
          : undefined,
      }))
    : [{ type: 'platform' as const }]; // Default to platform only

  return {
    id: node.id,
    type: 'hitl',
    title: String(config.title || 'Approval Required'),
    description: config.description ? String(config.description) : undefined,
    display: Array.isArray(config.display)
      ? config.display.map((item: { label: string; value: string }) => ({
          label: String(item.label || ''),
          value: String(item.value || ''),
        }))
      : undefined,
    inputs: Array.isArray(config.inputs)
      ? config.inputs.map((input: Record<string, unknown>) => ({
          id: String(input.id || ''),
          question: String(input.question || ''),
          input_type: (String(input.input_type || 'text') as HitlInputType),
          options: Array.isArray(input.options)
            ? input.options.map((opt: { value: string; label: string }) => ({
                value: String(opt.value || ''),
                label: String(opt.label || ''),
              }))
            : undefined,
          required: Boolean(input.required),
          placeholder: input.placeholder ? String(input.placeholder) : undefined,
          default_value: input.default_value as string | number | boolean | undefined,
        }))
      : [],
    timeout_seconds: typeof config.timeout_seconds === 'number'
      ? config.timeout_seconds
      : 86400,
    delivery_channels: deliveryChannels,
    ui: {
      position: node.position,
    },
  };
}

function createBrowserNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};

  // Build expect_outputs from output_schema (same pattern as tool block)
  const inlineSchema = buildInlineSchemaSpec(config.output_schema);
  const expectOutput = inlineSchema
    ? { mode: 'json' as const, schema: inlineSchema }
    : undefined;

  return {
    id: node.id,
    type: 'browser',
    task: String(config.task || ''),
    inputs: isRecord(config.inputs)
      ? (config.inputs as Record<string, JsonValue>)
      : {},
    browser_profile_id: config.browser_profile_id ? String(config.browser_profile_id) : null,
    max_steps: typeof config.max_steps === 'number' ? config.max_steps : 25,
    timeout_seconds: typeof config.timeout_seconds === 'number' ? config.timeout_seconds : 300,
    expect_outputs: expectOutput,
    save_screenshots: config.save_screenshots === true ? true : undefined,
    ui: {
      position: node.position,
    },
  };
}

function createImageGenNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const model = config.model || 'sourceful/riverflow-v2-fast';
  const prompt = config.prompt ? String(config.prompt) : '';
  const size = config.size || '1024x1024';

  const inputs: Record<string, JsonValue> = {
    model: String(model),
    prompt: String(prompt),
    size: String(size),
  };

  return {
    id: node.id,
    type: 'image_gen',
    inputs,
    ui: { position: node.position },
  };
}

function serializeAgentTools(config: Record<string, unknown>): JsonValue[] {
  const tools: JsonValue[] = [];
  const toolNames = Array.isArray(config.tools) ? config.tools : [];
  const toolConnections = isRecord(config.tool_connections) ? config.tool_connections : {};
  const toolResourceIds = isRecord(config.tool_resource_ids) ? config.tool_resource_ids : {};

  for (const toolName of toolNames) {
    const connectionId = toolConnections[toolName as string];
    const resourceId = toolResourceIds[toolName as string];
    if (connectionId != null || resourceId != null) {
      tools.push({
        name: String(toolName),
        ...(connectionId != null ? { connection_id: connectionId as number } : {}),
        ...(resourceId != null ? { integration_resource_id: resourceId as number } : {}),
      });
    } else {
      tools.push(String(toolName));
    }
  }
  return tools;
}

function createAgentNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const model = config.model || getUserDefaultModel();
  const prompt = config.prompt ? String(config.prompt) : '';
  const maxIterations = typeof config.max_iterations === 'number' ? config.max_iterations : 10;
  const memoryBankId =
    typeof config.memory_bank_id === 'string' && config.memory_bank_id.trim().length > 0
      ? config.memory_bank_id
      : null;

  const tools = serializeAgentTools(config);

  const enableArtifacts = config.enable_artifacts === true;

  // Backend expects model, prompt, tools, max_iterations nested inside inputs
  return {
    id: node.id,
    type: 'agent',
    inputs: {
      model: String(model),
      prompt: String(prompt),
      ...(memoryBankId ? { memory_bank_id: memoryBankId } : {}),
      tools,
      max_iterations: maxIterations,
      ...(enableArtifacts ? { enable_artifacts: true } : {}),
    },
    outputs: buildLlmOutputSchema(config),
    ui: { position: node.position },
  };
}

function createEABotNode(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const meetingUrl = config.meeting_url ? convertTemplateString(String(config.meeting_url)) : '';
  const botName = config.bot_name ? convertTemplateString(String(config.bot_name)) : 'meetingsEA';
  const recordingMode = String(config.recording_mode || 'speaker_view');
  const timeoutSeconds = typeof config.timeout_seconds === 'number' ? config.timeout_seconds : 14400;

  return {
    id: node.id,
    type: 'ea_bot',
    inputs: {
      meeting_url: meetingUrl,
      bot_name: botName,
      recording_mode: recordingMode,
    },
    timeout_seconds: timeoutSeconds,
    ui: { position: node.position },
  };
}

function buildLlmOutputSchema(config: Record<string, unknown>) {
  const inlineSchema = buildInlineSchemaSpec(config.output_schema);
  return inlineSchema ? { mode: 'json' as const, schema: inlineSchema } : { mode: 'text' as const };
}

function createIfNodeFlat(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const conditionRaw = String(config.condition ?? '').trim();

  // Note: With edge-based architecture, branches are defined by edges
  // (type=conditional_true, type=conditional_false), not nested node arrays.
  return {
    id: node.id,
    type: 'if',
    condition: conditionRaw,
    ui: {
      position: node.position,
    },
  };
}

function createForEachNodeFlat(node: FlowNode): WorkflowNode {
  const config = node.data?.config ?? {};
  const mode = config.array_mode === 'literal' ? 'literal' : 'variable';

  let items: string;
  if (mode === 'literal') {
    const literal = Array.isArray(config.array_literal) ? config.array_literal : [];
    items = JSON.stringify(literal);
  } else {
    const variableRef = String(config.array_variable ?? config.array_var ?? '').trim();
    items = variableRef;
  }

  // Note: With edge-based architecture, loop body is defined by edges
  // (type=loop_body for body, type=loop_exit for continuation), not nested node arrays.
  return {
    id: node.id,
    type: 'for_each',
    items,
    item_var: typeof config.item_var === 'string' ? config.item_var : 'item',
    index_var: typeof config.index_var === 'string' ? config.index_var : 'index',
    ui: {
      position: node.position,
    },
  };
}

// Legacy functions below are kept for backward compatibility but are no longer used
// in the edge-based architecture. They may be removed in a future cleanup.


function findNextNodeId(
  node: FlowNode,
  ctx: GraphContext,
  visited: Set<string>,
): string | undefined {
  const edges = ctx.outgoing.get(node.id) ?? [];

  if (node.data?.type === 'for_loop') {
    const exitEdge = edges.find((edge) => getBranch(edge) === 'exit' && !visited.has(edge.target));
    if (exitEdge) {
      return exitEdge.target;
    }
  }

  const defaultEdge = edges.find((edge) => !getBranch(edge) && !visited.has(edge.target));
  return defaultEdge?.target;
}


function compareNodesByTarget(
  a: WorkflowEdge,
  b: WorkflowEdge,
  nodeById: Map<string, FlowNode>,
): number {
  const nodeA = nodeById.get(a.target) ?? ({ position: { x: 0, y: 0 } } as FlowNode);
  const nodeB = nodeById.get(b.target) ?? ({ position: { x: 0, y: 0 } } as FlowNode);
  return compareNodes(nodeA, nodeB);
}

function compareNodesBySource(
  a: WorkflowEdge,
  b: WorkflowEdge,
  nodeById: Map<string, FlowNode>,
): number {
  const nodeA = nodeById.get(a.source) ?? ({ position: { x: 0, y: 0 } } as FlowNode);
  const nodeB = nodeById.get(b.source) ?? ({ position: { x: 0, y: 0 } } as FlowNode);
  return compareNodes(nodeA, nodeB);
}

function getNodeY(node?: FlowNode): number {
  return node?.position?.y ?? 0;
}

function getNodeX(node?: FlowNode): number {
  return node?.position?.x ?? 0;
}

function getNodeId(node?: FlowNode): string {
  return node?.id ?? '';
}

function compareNodes(a?: FlowNode, b?: FlowNode): number {
  const yDiff = getNodeY(a) - getNodeY(b);
  if (yDiff !== 0) return yDiff;

  const xDiff = getNodeX(a) - getNodeX(b);
  if (xDiff !== 0) return xDiff;

  return getNodeId(a).localeCompare(getNodeId(b));
}

function getBranch(edge: WorkflowEdge): BranchLabel | undefined {
  const branch = edge.data?.branch;
  if (branch === 'true' || branch === 'false' || branch === 'loop' || branch === 'exit') {
    return branch;
  }
  return undefined;
}

/**
 * Converts a ReactFlow edge to a backend workflow spec edge.
 * Maps frontend branch labels to backend EdgeType enum values.
 * For trigger edges, uses the 'trigger' edge type.
 */
function convertEdgeToSpecEdge(edge: WorkflowEdge, isTriggerEdge: boolean = false): WorkflowSpecEdge {
  // Build edge ui for UI data
  const ui: JsonObject = {};
  if (edge.sourceHandle !== null && edge.sourceHandle !== undefined) {
    ui.sourceHandle = edge.sourceHandle;
  }
  if (edge.targetHandle !== null && edge.targetHandle !== undefined) {
    ui.targetHandle = edge.targetHandle;
  }

  // If it's a trigger edge, use trigger type directly
  if (isTriggerEdge) {
    return {
      source: edge.source,  // trigger.key
      target: edge.target,  // node.id
      type: 'trigger',
      ...(Object.keys(ui).length > 0 && { ui }),
    };
  }

  // Otherwise, handle control flow branches as before
  const branch = edge.data?.branch;
  let edgeType: EdgeType = 'default';

  if (branch === 'true') {
    edgeType = 'conditional_true';
  } else if (branch === 'false') {
    edgeType = 'conditional_false';
  } else if (branch === 'loop') {
    edgeType = 'loop_body';
  } else if (branch === 'exit') {
    edgeType = 'loop_exit';
  }

  return {
    source: edge.source,
    target: edge.target,
    type: edgeType,
    ...(Object.keys(ui).length > 0 && { ui }),
  };
}

function buildGraphFromSpec(spec: WorkflowSpec): WorkflowGraphData {
  const nodes: FlowNode[] = [];
  const edges: WorkflowEdge[] = [];
  const defaultPositions = computeDefaultPositions(spec);

  // Build workflow nodes from spec, reading positions from ui
  spec.nodes.forEach((specNode, index) => {
    const nodeId = specNode.id || `node-${index}`;
    const position = specNode.ui?.position && typeof specNode.ui.position === 'object'
      ? (specNode.ui.position as { x: number; y: number })
      : defaultPositions.get(nodeId) ?? { x: 0, y: index * 200 };

    const flowNode: FlowNode = {
      id: nodeId,
      type: mapSpecNodeType(specNode.type),
      position,
      data: convertSpecNodeToData(specNode),
    };
    nodes.push(flowNode);
  });

  // Build trigger nodes with positions from ui_meta
  (spec.triggers ?? []).forEach((trigger, index) => {
    const position = trigger.ui_meta?.position && typeof trigger.ui_meta.position === 'object'
      ? (trigger.ui_meta.position as { x: number; y: number })
      : { x: index * 280, y: -150 };

    nodes.push({
      id: trigger.id,
      type: 'trigger',
      position,
      data: {
        type: 'trigger',
        label: trigger.id,
        config: {
          triggerKey: trigger.key,
        },
        triggerMeta: {
          handlers: {},
        },
      } as WorkflowNodeData,
    });
  });

  // Build edges from spec.edges, reading handles from ui
  (spec.edges ?? []).forEach((specEdge, index) => {
    const isTriggerEdge = specEdge.type === 'trigger';

    // Map edge type to branch label for UI
    let branch: BranchLabel | undefined;
    if (specEdge.type === 'conditional_true') branch = 'true';
    else if (specEdge.type === 'conditional_false') branch = 'false';
    else if (specEdge.type === 'loop_body') branch = 'loop';
    else if (specEdge.type === 'loop_exit') branch = 'exit';

    const edge: WorkflowEdge = {
      // Generate unique ID if not provided by backend
      id: (specEdge as { id?: string }).id ?? `${specEdge.source}-${specEdge.target}-${specEdge.type ?? 'default'}-${index}`,
      source: specEdge.source,
      target: specEdge.target,
      // IMPORTANT: sourceHandle must match branch for conditional/loop edges to render from correct handle
      // React Flow uses sourceHandle to determine which Handle component the edge visually connects from
      // Priority: computed branch (from edge type) > stored UI value > null
      sourceHandle: branch ?? (specEdge.ui?.sourceHandle as string | null) ?? null,
      targetHandle: (specEdge.ui?.targetHandle as string | null) ?? null,
      data: branch ? { branch } : undefined,
      ...(isTriggerEdge && {
        style: {
          stroke: 'hsl(239 84% 67%)',
          strokeWidth: 3,
          strokeDasharray: '5,5',
        },
      }),
    };
    edges.push(edge);
  });

  return { nodes, edges };
}

function computeDefaultPositions(spec: WorkflowSpec): Map<string, { x: number; y: number }> {
  const HORIZONTAL_SPACING = 220;
  const VERTICAL_SPACING = 100;

  const nodeIds = spec.nodes.map((n, idx) => n.id || `node-${idx}`);
  const incoming = new Map<string, Set<string>>();

  nodeIds.forEach((id) => incoming.set(id, new Set()));

  (spec.edges ?? []).forEach((edge) => {
    if (!incoming.has(edge.target)) return;
    incoming.get(edge.target)?.add(edge.source);
  });

  const depths = new Map<string, number>();

  const dfsDepth = (id: string, stack: Set<string>): number => {
    if (depths.has(id)) return depths.get(id)!;
    if (stack.has(id)) return 0;
    stack.add(id);
    const parents = incoming.get(id) ?? new Set();
    let depth = 0;
    parents.forEach((parent) => {
      if (!incoming.has(parent)) {
        depth = Math.max(depth, 0);
        return;
      }
      depth = Math.max(depth, dfsDepth(parent, stack) + 1);
    });
    stack.delete(id);
    depths.set(id, depth);
    return depth;
  };

  nodeIds.forEach((id) => dfsDepth(id, new Set()));

  const levels = new Map<number, string[]>();
  depths.forEach((depth, id) => {
    if (!levels.has(depth)) levels.set(depth, []);
    levels.get(depth)!.push(id);
  });

  const positions = new Map<string, { x: number; y: number }>();
  Array.from(levels.entries())
    .sort(([a], [b]) => a - b)
    .forEach(([depth, ids]) => {
      ids.sort();
      const offset = ((ids.length - 1) * HORIZONTAL_SPACING) / 2;
      ids.forEach((id, idx) => {
        positions.set(id, {
          x: idx * HORIZONTAL_SPACING - offset,
          y: depth * VERTICAL_SPACING,
        });
      });
    });

  return positions;
}

function mapSpecNodeType(specType: WorkflowNode['type']): WorkflowNodeData['type'] {
  switch (specType) {
    case 'tool':
      return 'tool';
    case 'if':
      return 'if_else';
    case 'for_each':
      return 'for_loop';
    case 'hitl':
      return 'hitl';
    case 'browser':
      return 'browser';
    case 'mcp':
      return 'mcp';
    case 'image_gen':
      return 'image_gen';
    case 'agent':
      return 'agent';
    case 'ea_bot':
      return 'ea_bot';
    default:
      return 'tool';
  }
}

function buildToolNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'tool') {
    throw new Error('Expected tool node');
  }

  const toolInputs = getSpecNodeInputs(specNode);
  const toolOutputSchema =
    specNode.expect_outputs?.mode === 'json' ? unwrapInlineSchema(specNode.expect_outputs.schema) : undefined;

  return {
    type: 'tool',
    label: specNode.id,
    config: {
      tool_name: specNode.tool,
      params: toolInputs ?? {},
      ...(toolOutputSchema && { output_schema: toolOutputSchema }),
      ...(specNode.connection_id != null && { connection_id: specNode.connection_id }),  // Multi-account OAuth support
    },
  };
}

function buildIfNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'if') {
    throw new Error('Expected if node');
  }

  return {
    type: 'if_else',
    label: specNode.id,
    config: {
      condition: specNode.condition,
    },
  };
}

function buildForEachNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'for_each') {
    throw new Error('Expected for_each node');
  }

  return {
    type: 'for_loop',
    label: specNode.id,
    config: {
      array_mode: 'variable',
      array_variable: specNode.items,
      item_var: specNode.item_var,
      index_var: specNode.index_var,
    },
  };
}

function buildHitlNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'hitl') {
    throw new Error('Expected hitl node');
  }

  return {
    type: 'hitl',
    label: specNode.id,
    config: {
      title: specNode.title,
      description: specNode.description,
      display: specNode.display?.map((item) => ({
        label: item.label,
        value: item.value,
      })),
      inputs: specNode.inputs?.map((input) => ({
        ...input,
      })),
      timeout_seconds: specNode.timeout_seconds,
      delivery_channels: specNode.delivery_channels?.map((channel) => ({
        type: channel.type,
        gmail: channel.gmail
          ? { recipient_email: channel.gmail.recipient_email }
          : undefined,
      })),
    },
  };
}

function buildBrowserNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'browser') {
    throw new Error('Expected browser node');
  }

  // Extract output_schema from expect_outputs (same pattern as tool block)
  const outputSchema =
    specNode.expect_outputs?.mode === 'json'
      ? unwrapInlineSchema(specNode.expect_outputs.schema)
      : undefined;

  return {
    type: 'browser',
    label: specNode.id,
    config: {
      task: specNode.task,
      inputs: specNode.inputs ?? {},
      browser_profile_id: specNode.browser_profile_id,
      max_steps: specNode.max_steps,
      timeout_seconds: specNode.timeout_seconds,
      output_schema: outputSchema,
      save_screenshots: specNode.save_screenshots,
    },
  };
}

function buildImageGenNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'image_gen') {
    throw new Error('Expected image_gen node');
  }

  const model = String(specNode.inputs.model ?? 'sourceful/riverflow-v2-fast');
  const prompt = String(specNode.inputs.prompt ?? '');
  const size = String(specNode.inputs.size ?? '1024x1024');

  return {
    type: 'image_gen',
    label: specNode.id,
    config: {
      model,
      prompt,
      size,
    },
  };
}

function deserializeMcpStdioFields(server: string, authObj: Record<string, unknown> | undefined) {
  const parts = server.split(' ');
  return {
    stdio_command: parts[0] || '',
    stdio_args: parts.slice(1),
    ...(authObj?.env && typeof authObj.env === 'object' ? { stdio_env: authObj.env as Record<string, string> } : {}),
  };
}

function deserializeMcpHttpFields(authObj: Record<string, unknown> | undefined) {
  if (authObj?.headers && typeof authObj.headers === 'object') {
    return { http_headers: authObj.headers as Record<string, string> };
  }
  return {};
}

function buildMcpNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'mcp') {
    throw new Error('Expected mcp node');
  }

  const serverType = specNode.server_type || 'http';
  const authObj = specNode.auth as Record<string, unknown> | undefined;

  const structuredFields = serverType === 'stdio' && specNode.server
    ? deserializeMcpStdioFields(specNode.server, authObj)
    : serverType === 'http'
      ? deserializeMcpHttpFields(authObj)
      : {};

  return {
    type: 'mcp',
    label: specNode.id,
    config: {
      server: specNode.server,
      server_type: serverType,
      tool: specNode.tool,
      inputs: specNode.inputs ?? {},
      auth: specNode.auth ?? undefined,
      ...(specNode.mcp_server_config_id ? { mcp_server_config_id: specNode.mcp_server_config_id } : {}),
      ...structuredFields,
    },
  };
}

function parseAgentTools(rawTools: (string | { name: string; connection_id?: number; integration_resource_id?: number })[]): {
  tools: string[];
  toolConnections: Record<string, number | null>;
  toolResourceIds: Record<string, number | null>;
} {
  const tools: string[] = [];
  const toolConnections: Record<string, number | null> = {};
  const toolResourceIds: Record<string, number | null> = {};

  for (const tool of rawTools) {
    if (typeof tool === 'string') {
      tools.push(tool);
    } else if (typeof tool === 'object' && tool.name) {
      tools.push(tool.name);
      if (tool.connection_id != null) {
        toolConnections[tool.name] = tool.connection_id;
      }
      if (tool.integration_resource_id != null) {
        toolResourceIds[tool.name] = tool.integration_resource_id;
      }
    }
  }

  return { tools, toolConnections, toolResourceIds };
}

function buildAgentNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'agent') {
    throw new Error('Expected agent node');
  }

  // Backend stores model, prompt, memory_bank_id, tools, max_iterations inside inputs
  const inputs = specNode.inputs || {};
  const model = String(inputs.model ?? getUserDefaultModel());
  const prompt = String(inputs.prompt ?? '');
  const memoryBankId =
    typeof inputs.memory_bank_id === 'string' && inputs.memory_bank_id.trim().length > 0
      ? inputs.memory_bank_id
      : null;
  const rawTools = inputs.tools as (string | { name: string; connection_id?: number; integration_resource_id?: number })[] ?? [];
  const maxIterations = typeof inputs.max_iterations === 'number' ? inputs.max_iterations : 10;
  const enableArtifacts = inputs.enable_artifacts === true;

  // Parse tools - can be strings or objects with connection_id / integration_resource_id
  const { tools, toolConnections, toolResourceIds } = parseAgentTools(rawTools);

  // Extract structured output schema from outputs field
  const agentOutputSchema =
    specNode.outputs?.mode === 'json' ? unwrapInlineSchema(specNode.outputs?.schema) : undefined;

  return {
    type: 'agent',
    label: specNode.id,
    config: {
      model,
      prompt,
      memory_bank_id: memoryBankId,
      tools,
      tool_connections: toolConnections,
      tool_resource_ids: toolResourceIds,
      max_iterations: maxIterations,
      ...(enableArtifacts && { enable_artifacts: true }),
      ...(agentOutputSchema && { output_schema: agentOutputSchema }),
    },
  };
}

function buildEABotNodeData(specNode: WorkflowNode): WorkflowNodeData {
  if (specNode.type !== 'ea_bot') {
    throw new Error('Expected ea_bot node');
  }

  const inputs = specNode.inputs || {};
  return {
    type: 'ea_bot',
    label: specNode.id,
    config: {
      meeting_url: convertTemplateString(String(inputs.meeting_url ?? ''), 'toBuilder'),
      bot_name: String(inputs.bot_name ?? 'meetingsEA'),
      recording_mode: String(inputs.recording_mode ?? 'speaker_view'),
      timeout_seconds: specNode.timeout_seconds ?? 14400,
    },
  };
}

function convertSpecNodeToData(specNode: WorkflowNode): WorkflowNodeData {
  switch (specNode.type) {
    case 'tool':
      return buildToolNodeData(specNode);
    case 'if':
      return buildIfNodeData(specNode);
    case 'for_each':
      return buildForEachNodeData(specNode);
    case 'hitl':
      return buildHitlNodeData(specNode);
    case 'browser':
      return buildBrowserNodeData(specNode);
    case 'image_gen':
      return buildImageGenNodeData(specNode);
    case 'agent':
      return buildAgentNodeData(specNode);
    case 'mcp':
      return buildMcpNodeData(specNode);
    case 'ea_bot':
      return buildEABotNodeData(specNode);
    default:
      return { type: 'tool', label: specNode.id, config: {} };
  }
}

function isRecord(value: unknown): value is Record<string, JsonValue> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function normalizeJsonSchema(schema: unknown): JsonObject | undefined {
  if (!isRecord(schema)) {
    return undefined;
  }
  if (Object.keys(schema).length === 0) {
    return undefined;
  }
  return schema as JsonObject;
}

function buildInlineSchemaSpec(schema: unknown): JsonObject | undefined {
  const jsonSchema = normalizeJsonSchema(schema);
  if (!jsonSchema) {
    return undefined;
  }
  return { json_schema: jsonSchema };
}

function unwrapInlineSchema(schemaSpec: unknown): JsonObject | undefined {
  if (!isRecord(schemaSpec)) {
    return undefined;
  }

  if (
    'json_schema' in schemaSpec &&
    isRecord((schemaSpec as Record<string, JsonValue>).json_schema)
  ) {
    return (schemaSpec as { json_schema: JsonObject }).json_schema;
  }

  if (
    'schema' in schemaSpec &&
    isRecord((schemaSpec as Record<string, JsonValue>).schema)
  ) {
    return (schemaSpec as { schema: JsonObject }).schema;
  }

  if (
    'id' in schemaSpec &&
    typeof (schemaSpec as Record<string, JsonValue>).id === 'string' &&
    Object.keys(schemaSpec).length === 1
  ) {
    return undefined;
  }

  return schemaSpec as JsonObject;
}


function getSpecNodeInputs(specNode: WorkflowNode): Record<string, JsonValue> | undefined {
  // Only Task/Tool/LLM nodes have inputs
  const withInputs = specNode as Partial<{ inputs: unknown; in: unknown; in_: unknown }>;

  // Support both new (inputs) and legacy (in, in_) formats for backward compatibility
  const rawInputs = (withInputs.inputs ?? withInputs.in ?? withInputs.in_);

  if (isRecord(rawInputs)) {
    return rawInputs as Record<string, JsonValue>;
  }
  return undefined;
}

function stripInputNodes(graph: WorkflowGraphData): WorkflowGraphData {
  const removedIds = new Set(
    graph.nodes
      .filter((node) => {
        const nodeType = (node.data?.type ?? node.type) as string | undefined;
        return nodeType === 'input';
      })
      .map((node) => node.id),
  );
  if (removedIds.size === 0) {
    return graph;
  }
  return {
    nodes: graph.nodes.filter((node) => !removedIds.has(node.id)),
    edges: graph.edges.filter(
      (edge) => !removedIds.has(edge.source) && !removedIds.has(edge.target),
    ),
  };
}

/**
 * Synchronizes trigger nodes on the canvas with fresh trigger specs from the triggers store.
 * This ensures that trigger nodes display up-to-date data (e.g., webhook URLs, signing secrets)
 * after operations like publishing that enrich trigger specs on the backend.
 *
 * @param nodes - Current canvas nodes
 * @param triggerSpecs - Fresh trigger specs from the triggers store
 * @returns Updated nodes array with synced trigger data
 */
export function syncTriggerNodesToStore(nodes: Node<WorkflowNodeData>[], triggerSpecs: TriggerSpec[]): Node<WorkflowNodeData>[] {
  return nodes.map(node => {
    // Only update trigger nodes
    if (node.type !== 'trigger' || !node.data.triggerMeta) return node;

    const freshTriggerSpec = triggerSpecs.find((trigger) => trigger.id === node.id);

    if (freshTriggerSpec) {
      // Update the subscription data with fresh spec from store
      return {
        ...node,
        data: {
          ...node.data,
          triggerMeta: {
            ...node.data.triggerMeta,
            subscription: freshTriggerSpec,
          },
        },
      };
    }
    return node;
  });
}
