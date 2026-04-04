/**
 * Block Configuration Panel
 *
 * Right sidebar panel for configuring selected block.
 * Supports editing parameters and OAuth scopes.
 */
import { Node } from '@xyflow/react';
import { Card, CardContent } from '@/components/ui/card';

import {
  ToolBlockSection,
  IfElseBlockSection,
  ForLoopBlockSection,
  MCPBlockSection,
  TriggerBlockSection,
  HITLBlockSection,
  BrowserBlockSection,
  ImageGenBlockSection,
  AgentBlockSection,
  EABotBlockSection,
  ToolMetadata,
  useTemplateAutocomplete,
} from '../block-config';
import { WorkflowEdge, WorkflowNodeData, WorkflowNodeUpdateOptions } from '../types';
import type { InputDef, JsonObject, TriggerSpec } from '@/types/workflow-spec';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import { useBlockConfigState } from './hooks/useBlockConfigState';
import { useToolSchema } from './hooks/useToolSchema';
import { useTemplateContext } from './hooks/useTemplateContext';
import { useLiveUpdate } from './hooks/useLiveUpdate';
import { useAutoSave } from './hooks/useAutoSave';
import { SaveButton } from './SaveButton';
import type { Dispatch, SetStateAction } from 'react';

interface BlockSectionProps {
  node: Node<WorkflowNodeData>;
  config: ToolBlockConfig;
  setConfig: Dispatch<SetStateAction<ToolBlockConfig>>;
  toolSchema?: ToolMetadata;
  templateAutocomplete: ReturnType<typeof useTemplateAutocomplete>;
  validationErrors: Record<string, string>;
  useStructuredOutput: boolean;
  setUseStructuredOutput: (value: boolean) => void;
  structuredOutputSchema?: JsonObject;
  onStructuredOutputSchemaChange: (schema?: JsonObject) => void;
  /** All nodes in workflow for variable discovery */
  allNodes?: Node<WorkflowNodeData>[];
  /** All edges in workflow for variable discovery */
  allEdges?: WorkflowEdge[];
  readOnly?: boolean;
}

// Helper component to render block-specific config sections
// eslint-disable-next-line max-lines-per-function
function BlockSection({
  node,
  config,
  setConfig,
  toolSchema,
  templateAutocomplete,
  validationErrors,
  useStructuredOutput,
  setUseStructuredOutput,
  structuredOutputSchema,
  onStructuredOutputSchemaChange,
  allNodes,
  allEdges,
  readOnly,
}: BlockSectionProps) {
  if (node.data.type === 'code') {
    return (
      <div className="text-sm text-muted-foreground">
        Code blocks are temporarily disabled while we migrate to the new workflow engine.
      </div>
    );
  }

  switch (node.data.type) {
    case 'tool':
      return (
        <ToolBlockSection
          config={config}
          setConfig={setConfig}
          toolSchema={toolSchema}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
          allNodes={allNodes}
          allEdges={allEdges}
          currentNode={node}
        />
      );
    case 'if_else':
      return (
        <IfElseBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'for_loop':
      return (
        <ForLoopBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'mcp':
      return (
        <MCPBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'hitl':
      return (
        <HITLBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'browser':
      return (
        <BrowserBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
          useStructuredOutput={useStructuredOutput}
          setUseStructuredOutput={setUseStructuredOutput}
          structuredOutputSchema={structuredOutputSchema}
          onStructuredOutputSchemaChange={onStructuredOutputSchemaChange}
        />
      );
    case 'image_gen':
      return (
        <ImageGenBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'agent':
      return (
        <AgentBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
          useStructuredOutput={useStructuredOutput}
          setUseStructuredOutput={setUseStructuredOutput}
          structuredOutputSchema={structuredOutputSchema}
          onStructuredOutputSchemaChange={onStructuredOutputSchemaChange}
        />
      );
    case 'ea_bot':
      return (
        <EABotBlockSection
          config={config}
          setConfig={setConfig}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
        />
      );
    case 'trigger':
      return (
        <TriggerBlockSection
          node={node}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
          readOnly={readOnly}
        />
      );
    default:
      return null;
  }
}

interface BlockConfigPanelProps {
  node: Node<WorkflowNodeData> | null;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
  allNodes?: Node<WorkflowNodeData>[]; // All nodes in workflow for reference dropdown
  allEdges?: WorkflowEdge[];
  autoSave?: boolean; // Enable auto-save on unmount (default: true for backward compatibility)
  variant?: 'panel' | 'inline';
  liveUpdate?: boolean;
  liveUpdateDelayMs?: number;
  workflowInputs?: Record<string, InputDef>;
  triggers?: TriggerSpec[];
  showSaveButton?: boolean; // Explicitly control save button visibility (default: auto-detect)
  validationErrors?: Record<string, string>; // Validation errors from parent
  onChange?: (config: ToolBlockConfig, oauthScope?: string) => void; // Notify parent of local changes (for button enable, not for parent state update)
  readOnly?: boolean; // Make the config panel read-only (e.g., for proposal preview)
}

// eslint-disable-next-line complexity
export function BlockConfigPanel({
  node,
  onUpdate,
  allNodes = [],
  allEdges = [],
  autoSave = true,
  variant = 'panel',
  liveUpdate = false,
  liveUpdateDelayMs = 350,
  workflowInputs,
  triggers,
  showSaveButton,
  validationErrors = {},
  onChange,
  readOnly = false,
}: BlockConfigPanelProps) {
  const {
    config,
    setConfig,
    oauthScope,
    inputRefs,
    setInputRefs,
    useStructuredOutput,
    setUseStructuredOutput,
    structuredOutputSchema,
    handleStructuredOutputSchemaChange,
    configRef,
    isSavingRef,
    originalNodeRef,
  } = useBlockConfigState({ node, onChange });

  const toolSchema = useToolSchema({ toolName: config.tool_name || config.toolName || '', node });
  const { templateAutocomplete } = useTemplateContext({ allNodes, allEdges, node, workflowInputs, triggers });
  useLiveUpdate({
    enabled: liveUpdate,
    delayMs: liveUpdateDelayMs,
    node,
    currentConfig: config,
    currentInputRefs: inputRefs,
    currentOauthScope: oauthScope,
    configRef,
    onUpdate,
  });
  useAutoSave({ enabled: autoSave, node, originalNodeRef, isSavingRef, configRef, onUpdate });

  if (!node) {
    return <div data-testid="block-config-panel" className="p-4 text-center text-muted-foreground">Select a block to configure</div>;
  }

  const shouldShowSaveButton = showSaveButton ?? (!autoSave && !liveUpdate);
  const content = (
    <>
      {readOnly && (
        <div className="mb-3 p-2 bg-sky-900/10 border border-sky-900/20 rounded-md">
          <p className="text-xs text-sky-900 dark:text-sky-100 font-medium">
            Preview Mode - Configuration is read-only
          </p>
        </div>
      )}
      <div className={readOnly ? 'pointer-events-none opacity-70' : ''}>
        <BlockSection
          node={node}
          config={config}
          setConfig={setConfig}
          toolSchema={toolSchema}
          templateAutocomplete={templateAutocomplete}
          validationErrors={validationErrors}
          useStructuredOutput={useStructuredOutput}
          setUseStructuredOutput={setUseStructuredOutput}
          structuredOutputSchema={structuredOutputSchema}
          onStructuredOutputSchemaChange={handleStructuredOutputSchemaChange}
          allNodes={allNodes}
          allEdges={allEdges}
          readOnly={readOnly}
        />
      </div>
      {!readOnly && (
        <SaveButton
          visible={shouldShowSaveButton}
          node={node}
          toolSchema={toolSchema}
          config={config}
          onUpdate={onUpdate}
        />
      )}
    </>
  );

  if (variant === 'inline') {
    return <div data-testid="block-config-panel" className="space-y-2 w-full">{content}</div>;
  }

  return (
    <Card data-testid="block-config-panel" className="w-full">
      <CardContent className="p-4 space-y-2 w-full">{content}</CardContent>
    </Card>
  );
}
