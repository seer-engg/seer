import { useState, useEffect, useLayoutEffect, useRef, useCallback, type Dispatch, type SetStateAction } from 'react';
import { Node } from '@xyflow/react';
import { WorkflowNodeData } from '../../types';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import type { JsonObject } from '@/types/workflow-spec';

interface UseBlockConfigStateParams {
  node: Node<WorkflowNodeData> | null;
  onChange?: (config: ToolBlockConfig, oauthScope?: string) => void;
}

export interface UseBlockConfigStateResult {
  // State
  config: ToolBlockConfig;
  setConfig: Dispatch<SetStateAction<ToolBlockConfig>>;
  oauthScope: string | undefined;
  setOAuthScope: (scope: string | undefined) => void;
  inputRefs: Record<string, string>;
  setInputRefs: Dispatch<SetStateAction<Record<string, string>>>;
  useStructuredOutput: boolean;
  setUseStructuredOutput: (use: boolean) => void;
  structuredOutputSchema: JsonObject | undefined;
  handleStructuredOutputSchemaChange: (schema?: JsonObject) => void;

  // Refs
  configRef: React.MutableRefObject<ToolBlockConfig>;
  inputRefsRef: React.MutableRefObject<Record<string, string>>;
  oauthScopeRef: React.MutableRefObject<string | undefined>;
  isSavingRef: React.MutableRefObject<boolean>;
  originalNodeRef: React.MutableRefObject<Node<WorkflowNodeData> | null>;
  lastSyncedNodeStateRef: React.MutableRefObject<{ nodeId: string | null; signature: string }>;
}

// Helper function to notify parent of config changes
function notifyConfigChange(params: {
  node: Node<WorkflowNodeData> | null;
  config: ToolBlockConfig;
  inputRefs: Record<string, string>;
  oauthScope: string | undefined;
  lastSyncedNodeId: string | null;
  onChange?: (config: ToolBlockConfig, oauthScope?: string) => void;
}): void {
  const { node, config, inputRefs, oauthScope, lastSyncedNodeId, onChange } = params;

  if (!onChange || !node) return;

  // Only notify if we've synced the node already (avoid triggering on initial load)
  if (lastSyncedNodeId !== node.id) {
    return;
  }

  // Include inputRefs in the config so dialog can detect changes to template variables
  const mergedConfig = {
    ...config,
    input_refs: inputRefs,
  };

  onChange(mergedConfig, oauthScope);
}

// Helper function to initialize/reset node sync state
function initializeNodeSyncState(params: {
  node: Node<WorkflowNodeData> | null;
  lastSyncedNodeStateRef: React.MutableRefObject<{ nodeId: string | null; signature: string }>;
}): void {
  const { node, lastSyncedNodeStateRef } = params;

  if (node) {
    // Only set nodeId if it's a different node
    if (lastSyncedNodeStateRef.current.nodeId !== node.id) {
      lastSyncedNodeStateRef.current.nodeId = node.id;
      // Keep existing signature to trigger sync effect
      // (signature will be updated by the main sync effect)
    }
  } else {
    // Reset when no node
    lastSyncedNodeStateRef.current = {
      nodeId: null,
      signature: '',
    };
  }
}

export function useBlockConfigState({
  node,
  onChange,
}: UseBlockConfigStateParams): UseBlockConfigStateResult {
  const [config, setConfig] = useState<ToolBlockConfig>({});
  const [oauthScope, setOAuthScope] = useState<string | undefined>();
  const [inputRefs, setInputRefs] = useState<Record<string, string>>({});
  const [useStructuredOutput, setUseStructuredOutput] = useState(false);
  const [structuredOutputSchema, setStructuredOutputSchema] = useState<JsonObject | undefined>();

  const lastSyncedNodeStateRef = useRef<{ nodeId: string | null; signature: string }>({
    nodeId: null,
    signature: '',
  });

  // Use refs to track latest values for auto-save on unmount
  const configRef = useRef(config);
  const inputRefsRef = useRef(inputRefs);
  const oauthScopeRef = useRef(oauthScope);
  const isSavingRef = useRef(false);
  const originalNodeRef = useRef(node);

  // Update refs synchronously when state changes to prevent stale reads in other effects
  useLayoutEffect(() => {
    configRef.current = config;
    inputRefsRef.current = inputRefs;
    oauthScopeRef.current = oauthScope;
  }, [config, inputRefs, oauthScope]);

  // Notify parent of local state changes (for change detection only, doesn't trigger parent state update)
  useEffect(() => {
    notifyConfigChange({
      node,
      config,
      inputRefs,
      oauthScope,
      lastSyncedNodeId: lastSyncedNodeStateRef.current.nodeId,
      onChange,
    });
  }, [config, inputRefs, oauthScope, onChange, node]);

  // Initialize lastSyncedNodeStateRef.nodeId immediately when node changes
  // This ensures onChange can fire even if user edits before sync completes
  useEffect(() => {
    initializeNodeSyncState({ node, lastSyncedNodeStateRef });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node?.id]); // Only run when node ID changes, not on every node property change

  // Sync node config to local state
  useEffect(() => {
    if (!node) {
      return;
    }

    const nodeConfig = node.data.config || {};
    const signature = JSON.stringify({
      config: nodeConfig,
      oauth_scope: node.data.oauth_scope,
      input_refs: node.data.config?.input_refs || {},
      // Include fields in signature to ensure sync when fields change
      fields: nodeConfig.fields,
    });

    if (
      lastSyncedNodeStateRef.current.nodeId === node.id &&
      lastSyncedNodeStateRef.current.signature === signature
    ) {
      return;
    }

    lastSyncedNodeStateRef.current = {
      nodeId: node.id,
      signature,
    };

    configRef.current = nodeConfig;
    inputRefsRef.current = node.data.config?.input_refs || {};
    oauthScopeRef.current = node.data.oauth_scope;
    setConfig(nodeConfig);
    setOAuthScope(node.data.oauth_scope);
    setInputRefs(node.data.config?.input_refs || {});
    setUseStructuredOutput(!!nodeConfig.output_schema);
    setStructuredOutputSchema(nodeConfig.output_schema);
    originalNodeRef.current = node; // Update original node reference
    // Only re-sync when switching to a different node (by ID), not on every node reference change.
    // The signature comparison above already handles content changes within the same node.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [node?.id]);

  const handleStructuredOutputSchemaChange = useCallback(
    (schema?: JsonObject) => {
      setStructuredOutputSchema(schema);
      setConfig(prev => {
        if (schema) {
          return { ...prev, output_schema: schema };
        }
        const next = { ...prev };
        delete next.output_schema;
        return next;
      });
    },
    [],
  );

  return {
    config,
    setConfig,
    oauthScope,
    setOAuthScope,
    inputRefs,
    setInputRefs,
    useStructuredOutput,
    setUseStructuredOutput,
    structuredOutputSchema,
    handleStructuredOutputSchemaChange,
    configRef,
    inputRefsRef,
    oauthScopeRef,
    isSavingRef,
    originalNodeRef,
    lastSyncedNodeStateRef,
  };
}
