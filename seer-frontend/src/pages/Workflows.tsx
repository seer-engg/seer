import { useMemo, useEffect, useCallback, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import type { TriggerWithConnection } from '@/components/workflows/dialogs/RunOptionsDialog';
import { TRIGGER_BROWSING_MODE } from '@/components/workflows/dialogs/TriggerEventPicker/types';
import type { TriggerEventItem } from '@/types/triggers';
import { generateTriggerEvent } from '@/lib/api-client';
import { WorkflowDialogs } from '@/components/workflows/layout/WorkflowDialogs';
import { CanvasSection, LeftChatPanel } from '@/components/workflows/layout/WorkflowPanels';
import { ContextualPane } from '@/components/workflows/panels/ContextualPane';
import {
  usePreviewGraph, useTriggerHandlersConfig,
  buildCanvasProps,
  getProviderFromTriggerKey, parseConnectionId,
} from '@/components/workflows/layout/workflowPanelHelpers';
import { useWorkflowHooks, useWorkflowInitialization } from '@/hooks/useWorkflowPageComposition';
import { useDebouncedAutosave } from '@/hooks/useDebouncedAutosave';
import { ResizablePanelGroup, ResizableHandle } from '@/components/ui/resizable';
import { useKeyboardShortcut } from '@/hooks/utility/useKeyboardShortcuts';
import { useWorkflowSync } from '@/hooks/useWorkflowSync';
import { useEnsureWorkflowSelected } from '@/hooks/useEnsureWorkflowSelected';
import { useWorkflowActions } from '@/hooks/useWorkflowActions';
import { useNodeActions } from '@/hooks/useNodeActions';
import { useTriggerHandlers } from '@/hooks/useTriggerHandlers';
import { useWorkflowPageState } from '@/hooks/useWorkflowPageState';
import { useWorkflowLifecycleEffects } from '@/hooks/useWorkflowLifecycleEffects';
import { useWorkflowCollaboration } from '@/hooks/useWorkflowCollaboration';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useSubscriptionHydration } from '@/hooks/useSubscriptionHydration';
import { useToolingData } from '@/hooks/useToolingData';
import { useTriggerMetaHydration } from '@/hooks/useTriggerMetaHydration';
import { useCanvasStore } from '@/stores';
import { useTriggersStore } from '@/stores/triggersStore';


// eslint-disable-next-line max-lines-per-function -- Core workflow page composition
export default function Workflows() {
  const state = useWorkflowPageState();
  const {
    urlWorkflowId, nodes, edges, isDirty, setNodes, setEdges, setSelectedNodeId,
    autosaveStatus, isImportDialogOpen, isKeymapOpen,
    isInputDialogOpen, proposalPreview,
    setImportDialogOpen, setKeymapOpen, setInputDialogOpen, setProposalPreview, setWorkflowName,
    selectedWorkflowId, setSelectedWorkflowId, inputData, setInputData, isLoadingWorkflow,
    setIsLoadingWorkflow, loadedWorkflow, workflowError, workflowListError, workflows, workflowsLoaded, isLoadingWorkflows, isFetchingWorkflow,
    isExecuting, isPublishing, isRestoringVersion, functionBlocksReady, functionBlockSchemas, functionBlocksMap,
    // Run options dialog state
    isRunOptionsDialogOpen, setRunOptionsDialogOpen,
    isTriggerEventPickerOpen, setTriggerEventPickerOpen,
    selectedTriggerForPicker, setSelectedTriggerForPicker,
  } = state;
  const activeWorkflowId = urlWorkflowId ?? selectedWorkflowId;

  useKeyboardShortcut({
    key: '/', modifiers: { ctrl: true, meta: true }, handler: () => setKeymapOpen(true),
    category: 'Help', description: 'Show keyboard shortcuts', scope: 'global',
  });

  const { workflowInputsDef, inputFields, gmailToolNames, gmailConnectionId, discordToolNames, discordConnectionId, slackToolNames, slackConnectionId, calendarToolNames, calendarConnectionId } = useWorkflowInitialization();

  const {
    handleExecute, handleExecuteWithInput, handlePublish: baseHandlePublish, handleRestoreVersion: baseHandleRestoreVersion,
    handleDeleteWorkflow: baseHandleDeleteWorkflow, handleRenameWorkflow: baseHandleRenameWorkflow,
    handleImportWorkflow, autosaveCallback, lifecycleStatus,
    lastRunVersionId, setLastRunVersionId, canRun, canPublish, runDisabledReason,
    publishDisabledReason, resetSavedDataRef,
    runWorkflowWithTriggerEvent,
  } = useWorkflowActions();

  const { triggerSave, saveImmediately, resetSavedData } = useDebouncedAutosave({
    data: { nodes, edges },
    onSave: autosaveCallback,
    options: { delay: 1000, enabled: !!activeWorkflowId && !isLoadingWorkflow },
  });

  const {
    workflowVersions, isLoadingWorkflowVersions, triggerCatalog,
    workflowTriggers, updateTrigger, removeTrigger, handleWorkflowGraphSync,
  } = useWorkflowHooks({
    selectedWorkflowId: activeWorkflowId, functionBlockSchemas, functionBlocksMap, setNodes, setEdges, setProposalPreview,
  });

  const { isConnectingGmail, handleConnectGmail, isConnectingDiscord, handleConnectDiscord, isConnectingSlack, handleConnectSlack, isConnectingCalendar, handleConnectCalendar } = useTriggerHandlers({
    gmailToolNames,
    discordToolNames,
    slackToolNames,
    calendarToolNames,
  });

  const {
    handleNodeDrop, handleCanvasNodeClick,
    handleNodeConfigUpdate,
  } = useNodeActions({
    autosaveCallback, triggerSave, gmailConnectionId,
    handleConnectGmail, isConnectingGmail, discordConnectionId,
    handleConnectDiscord, isConnectingDiscord,
  });

  useEffect(() => {
    resetSavedDataRef.current = resetSavedData;
  }, [resetSavedData, resetSavedDataRef]);

  const previewGraph = usePreviewGraph(proposalPreview, functionBlocksMap);
  const isPreviewActive = Boolean(previewGraph);

  const stableUpdateTrigger = useCallback(
    (triggerId: string, payload: Record<string, unknown>) =>
      activeWorkflowId
        ? Promise.resolve(updateTrigger(activeWorkflowId, triggerId, payload)).then(() => undefined)
        : Promise.resolve(),
    [activeWorkflowId, updateTrigger],
  );

  const stableToggleTrigger = useCallback(
    (triggerId: string, enabled: boolean) =>
      activeWorkflowId
        ? Promise.resolve(updateTrigger(activeWorkflowId, triggerId, { enabled })).then(() => undefined)
        : Promise.resolve(),
    [activeWorkflowId, updateTrigger],
  );

  const stableDeleteTrigger = useCallback(
    (triggerId: string) => {
      if (!activeWorkflowId) return Promise.resolve();
      // Remove from triggersStore (triggers autosave via markDirty)
      removeTrigger(activeWorkflowId, triggerId);
      // Remove the corresponding canvas node (triggerId equals node.id)
      useCanvasStore.getState().deleteNode(triggerId);
      return Promise.resolve();
    },
    [activeWorkflowId, removeTrigger],
  );

  const triggerHandlers = useTriggerHandlersConfig({
    updateTrigger: stableUpdateTrigger,
    toggleTrigger: stableToggleTrigger,
    deleteTrigger: stableDeleteTrigger,
  });

  useWorkflowLifecycleEffects({
    autosaveStatus, nodes, workflowId: activeWorkflowId, isLoadingWorkflow,
    isPreviewActive, isDirty, triggerSave, setNodes, setSelectedNodeId,
    triggerHandlers,
  });

  useSubscriptionHydration(activeWorkflowId, nodes, setNodes);

  // Handle account selection for multi-account OAuth triggers
  const handleTriggerAccountSelect = useCallback((triggerId: string, connectionId: number | null) => {
    if (!activeWorkflowId) return;
    // Get current trigger to preserve existing provider_config
    const triggers = useTriggersStore.getState().workflowTriggers.get(activeWorkflowId) ?? [];
    const trigger = triggers.find(t => t.id === triggerId);
    const currentProviderConfig = trigger?.provider_config ?? {};
    // Update provider_connection_id in provider_config
    updateTrigger(activeWorkflowId, triggerId, {
      provider_config: { ...currentProviderConfig, provider_connection_id: connectionId },
    });
  }, [activeWorkflowId, updateTrigger]);

  // Hydrate trigger metadata (descriptor, integration) for trigger nodes loaded from saved workflows
  // This ensures trigger nodes have full config after page refresh (both connection status AND config forms)
  const integrationContext = useMemo(() => ({
    gmailConnectionId,
    handleConnectGmail,
    isConnectingGmail,
    discordConnectionId,
    handleConnectDiscord,
    isConnectingDiscord,
    slackConnectionId,
    handleConnectSlack,
    isConnectingSlack,
    calendarConnectionId,
    handleConnectCalendar,
    isConnectingCalendar,
    onSelectAccount: handleTriggerAccountSelect,
  }), [
    gmailConnectionId,
    handleConnectGmail,
    isConnectingGmail,
    discordConnectionId,
    handleConnectDiscord,
    isConnectingDiscord,
    slackConnectionId,
    handleConnectSlack,
    isConnectingSlack,
    calendarConnectionId,
    handleConnectCalendar,
    isConnectingCalendar,
    handleTriggerAccountSelect,
  ]);
  useTriggerMetaHydration(nodes, setNodes, integrationContext);

  const selectedNodeId = useCanvasStore((state) => state.selectedNodeId);
  const { getConnectionId } = useToolingData();
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    // In preview mode, look for the node in previewGraph, otherwise use regular nodes
    const nodesToSearch = isPreviewActive && previewGraph ? previewGraph.nodes : nodes;
    return nodesToSearch.find((n) => n.id === selectedNodeId) ?? null;
  }, [selectedNodeId, nodes, isPreviewActive, previewGraph]);

  useWorkflowSync({
    urlWorkflowId, loadedWorkflow, workflowError, isFetchingWorkflow, isDirty, functionBlocksReady, functionBlocksMap, setSelectedWorkflowId,
    setWorkflowName, setNodes, setEdges, setProposalPreview, setLastRunVersionId, setIsLoadingWorkflow,
  });

  useEnsureWorkflowSelected({
    urlWorkflowId,
    workflows,
    workflowsLoaded,
    isLoading: isLoadingWorkflows,
    workflowListError,
  });

  // Use URL workflow ID when available to avoid race conditions during initial load
  const effectiveWorkflowId = activeWorkflowId;
  const {
    isReadOnly: isWorkflowReadOnly,
    lockMessage,
    acquireLock,
  } = useWorkflowCollaboration(effectiveWorkflowId);

  const handleRetryLock = useCallback(() => {
    void acquireLock();
  }, [acquireLock]);

  const handleRenameWorkflow = useCallback(async (workflowId: string, newName: string) => {
    if (isWorkflowReadOnly) return;
    await baseHandleRenameWorkflow(workflowId, newName);
  }, [baseHandleRenameWorkflow, isWorkflowReadOnly]);

  const handleDeleteWorkflow = useCallback(async (workflowId: string) => {
    if (isWorkflowReadOnly) return;
    await baseHandleDeleteWorkflow(workflowId);
  }, [baseHandleDeleteWorkflow, isWorkflowReadOnly]);

  const handleRestoreVersion = useCallback(async (versionId: number) => {
    if (isWorkflowReadOnly) return;
    await baseHandleRestoreVersion(versionId);
  }, [baseHandleRestoreVersion, isWorkflowReadOnly]);

  const handleNodeConfigUpdateWithLock = useCallback((
    nodeId: string,
    updates: Parameters<typeof handleNodeConfigUpdate>[1],
    options?: Parameters<typeof handleNodeConfigUpdate>[2],
  ) => {
    if (isWorkflowReadOnly) return;
    return handleNodeConfigUpdate(nodeId, updates, options);
  }, [handleNodeConfigUpdate, isWorkflowReadOnly]);

  // Extract subscription IDs from trigger nodes into a stable map.
  // Only produces a new reference when subscription IDs actually change,
  // preventing triggersWithConnection from re-deriving on every node
  // drag/select/position update.
  const triggerSubscriptionIdsRef = useRef(new Map<string, number | null>());
  const triggerSubscriptionIds = useMemo(() => {
    const next = new Map<string, number | null>();
    for (const node of nodes) {
      if (node.data.type !== 'trigger') continue;
      const uiMeta = node.data?.triggerMeta?.subscription?.ui_meta as Record<string, unknown> | undefined;
      next.set(node.id, (uiMeta?.subscription_id as number) ?? null);
    }
    // Return previous reference if contents unchanged
    const prev = triggerSubscriptionIdsRef.current;
    if (prev.size === next.size) {
      let same = true;
      for (const [k, v] of next) {
        if (prev.get(k) !== v) { same = false; break; }
      }
      if (same) return prev;
    }
    triggerSubscriptionIdsRef.current = next;
    return next;
  }, [nodes]);

  // Derive triggers with connection status for RunOptionsDialog
  // Uses is_connected from trigger catalog (backend is source of truth)
  // Falls back to toolsStore lookup for OAuth-based triggers
  const triggersWithConnection: TriggerWithConnection[] = useMemo(() => {
    return workflowTriggers.map((trigger) => {
      const provider = getProviderFromTriggerKey(trigger.key);
      const browsingMode = TRIGGER_BROWSING_MODE[trigger.key] ?? null;

      // Look up the trigger in the catalog to get backend-provided connection status
      const catalogEntry = triggerCatalog.find((t) => t.key === trigger.key);

      // Get subscription ID from stable map (avoids depending on full nodes array)
      const subscriptionId = triggerSubscriptionIds.get(trigger.id) ?? null;

      // For polling triggers, need OAuth connection
      if (browsingMode === 'polling') {
        const integrationType = provider === 'google' ? 'gmail' : provider;
        const connectionIdRaw = getConnectionId(integrationType);
        const connectionId = parseConnectionId(connectionIdRaw);
        const isConnected = catalogEntry?.is_connected ?? connectionId !== null;

        return {
          ...trigger,
          provider,
          connectionId,
          subscriptionId,
          isConnected,
          hasEvents: isConnected, // For polling, assume connected = can browse events
          browsingMode,
        } as TriggerWithConnection;
      }

      // For persisted triggers (webhooks, forms), check if subscription has events
      // Note: We'll need to query event counts asynchronously - for now default to true if subscription exists
      if (browsingMode === 'persisted') {
        const hasSubscription = subscriptionId !== null;

        return {
          ...trigger,
          provider,
          connectionId: null,
          subscriptionId,
          isConnected: hasSubscription, // Persisted triggers don't need OAuth connection
          hasEvents: hasSubscription, // Will be refined with actual event count query
          browsingMode,
        } as TriggerWithConnection;
      }

      // Cron and other triggers without event browsing
      return {
        ...trigger,
        provider,
        connectionId: null,
        subscriptionId,
        isConnected: true, // Cron triggers are always "connected"
        hasEvents: false,
        browsingMode,
      } as TriggerWithConnection;
    });
  }, [workflowTriggers, triggerCatalog, getConnectionId, triggerSubscriptionIds]);

  // Handler for selecting a trigger event
  const handleSelectTriggerForRealEvent = useCallback((trigger: TriggerWithConnection) => {
    // For polling triggers, need connectionId
    // For persisted triggers, need subscriptionId
    if (trigger.browsingMode === 'polling' && trigger.connectionId === null) return;
    if (trigger.browsingMode === 'persisted' && trigger.subscriptionId === null) return;

    setSelectedTriggerForPicker({
      triggerId: trigger.id,
      triggerKey: trigger.key,
      provider: trigger.provider,
      connectionId: trigger.connectionId ?? undefined,
      subscriptionId: trigger.subscriptionId ?? undefined,
      filterParams: trigger.provider_config ?? undefined,
    });
    setRunOptionsDialogOpen(false);
    setTriggerEventPickerOpen(true);
  }, [setSelectedTriggerForPicker, setRunOptionsDialogOpen, setTriggerEventPickerOpen]);

  const handleTriggerEventSelect = useCallback(async (event: TriggerEventItem) => {
    if (!selectedTriggerForPicker) return;
    setTriggerEventPickerOpen(false);
    setSelectedTriggerForPicker(null);
    await runWorkflowWithTriggerEvent(selectedTriggerForPicker.triggerId, event.envelope);
  }, [selectedTriggerForPicker, setTriggerEventPickerOpen, setSelectedTriggerForPicker, runWorkflowWithTriggerEvent]);

  const handleConnectProvider = useCallback((provider: string) => {
    if (provider === 'google') {
      handleConnectGmail();
    } else if (provider === 'discord') {
      handleConnectDiscord();
    }
  }, [handleConnectGmail, handleConnectDiscord]);

  // State for instant trigger execution
  const [isTriggering, setIsTriggering] = useState(false);

  // Handler for instant triggers (e.g., cron) - triggers workflow immediately with synthetic event
  const handleTriggerNow = useCallback(async (trigger: TriggerWithConnection) => {
    if (!activeWorkflowId) {
      toast.error('Please save the workflow first');
      return;
    }

    setIsTriggering(true);
    try {
      // Get subscription from triggersStore to access provider_config
      const triggers = useTriggersStore.getState().workflowTriggers.get(activeWorkflowId) ?? [];
      const triggerSpec = triggers.find((t) => t.id === trigger.id);
      const providerConfig = triggerSpec?.provider_config ?? {};

      // Generate synthetic event for this trigger
      const { envelope } = await generateTriggerEvent(
        trigger.key,
        trigger.id,
        providerConfig,
      );

      // Close dialog before running
      setRunOptionsDialogOpen(false);

      // Run workflow with the generated event
      await runWorkflowWithTriggerEvent(trigger.id, envelope);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to trigger workflow';
      toast.error(message);
    } finally {
      setIsTriggering(false);
    }
  }, [activeWorkflowId, setRunOptionsDialogOpen, runWorkflowWithTriggerEvent]);

  const handleRunClick = useCallback(() => {
    const hasTriggers = workflowTriggers.length > 0;
    if (hasTriggers) {
      // Instant triggers (e.g., cron) can trigger immediately
      const instantTriggers = triggersWithConnection.filter((t) => t.browsingMode === 'instant');

      // Browsable triggers need event selection
      const browsableTriggers = triggersWithConnection.filter((t) => {
        if (t.browsingMode === null || t.browsingMode === 'instant') return false;
        if (t.browsingMode === 'polling') return t.isConnected;
        if (t.browsingMode === 'persisted') return t.hasEvents;
        return false;
      });

      // If only one instant trigger and no browsable triggers, trigger immediately
      if (instantTriggers.length === 1 && browsableTriggers.length === 0) {
        handleTriggerNow(instantTriggers[0]);
        return;
      }

      // If only one browsable trigger and no instant triggers, go to event selection
      if (browsableTriggers.length === 1 && instantTriggers.length === 0) {
        handleSelectTriggerForRealEvent(browsableTriggers[0]);
        return;
      }

      // Multiple triggers or mixed types - show dialog
      if (instantTriggers.length > 0 || browsableTriggers.length > 0) {
        setRunOptionsDialogOpen(true);
        return;
      }

      // No usable triggers - still show dialog to guide user
      setRunOptionsDialogOpen(true);
      return;
    }
    const result = handleExecute(inputFields);
    if (result?.needsInput) {
      setInputDialogOpen(true);
    }
  }, [workflowTriggers, triggersWithConnection, handleSelectTriggerForRealEvent,
      handleTriggerNow, handleExecute, inputFields, setInputDialogOpen, setRunOptionsDialogOpen]);

  // Flush pending autosave before activating so the backend DRAFT reflects the canvas
  const handlePublishWithFlush = useCallback(async () => {
    if (isWorkflowReadOnly) return;
    await saveImmediately();
    await baseHandlePublish();
  }, [baseHandlePublish, isWorkflowReadOnly, saveImmediately]);

  const publishDisabledReasonWithLock = isWorkflowReadOnly
    ? lockMessage ?? 'You need the edit lock to activate this workflow.'
    : publishDisabledReason;
  const versionRestoreDisabledReason = isWorkflowReadOnly
    ? lockMessage ?? 'You need the edit lock to restore versions.'
    : effectiveWorkflowId
      ? undefined
      : 'Save the workflow before restoring versions';

  const canvasProps = useMemo(() => buildCanvasProps({
    selectedWorkflowId: effectiveWorkflowId, previewGraph, proposalPreview, lifecycleStatus,
    workflowVersions, isLoadingWorkflowVersions, isExecuting, isPublishing, isRestoringVersion,
    canRun, canPublish: canPublish && !isWorkflowReadOnly, publishDisabledReason: publishDisabledReasonWithLock,
    runDisabledReason, isImportDialogOpen,
    handleCanvasNodeClick, handleNodeDrop, handleRunClick, handlePublish: handlePublishWithFlush, handleRestoreVersion,
    setImportDialogOpen, handleImportWorkflow,
    readOnly: isWorkflowReadOnly,
    readOnlyReason: lockMessage,
    onRetryLock: handleRetryLock,
    versionRestoreDisabledReason,
  }), [effectiveWorkflowId, previewGraph, proposalPreview, lifecycleStatus,
    workflowVersions, isLoadingWorkflowVersions, isExecuting, isPublishing, isRestoringVersion,
    canRun, canPublish, publishDisabledReasonWithLock, runDisabledReason, isImportDialogOpen,
    handleCanvasNodeClick, handleNodeDrop, handleRunClick, handlePublishWithFlush, handleRestoreVersion,
    setImportDialogOpen, handleImportWorkflow, isWorkflowReadOnly, lockMessage, handleRetryLock,
    versionRestoreDisabledReason]);

  const allNodes = isPreviewActive && previewGraph ? previewGraph.nodes : nodes;
  const allEdges = isPreviewActive && previewGraph ? previewGraph.edges : edges;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3, ease: 'easeInOut' }}
      className="flex flex-col h-full bg-background"
    >
      <ResizablePanelGroup direction="horizontal" className="flex-1 overflow-hidden">
        <LeftChatPanel
          workflowId={effectiveWorkflowId}
          workflowName={loadedWorkflow?.name ?? 'Untitled Workflow'}
          onWorkflowGraphSync={handleWorkflowGraphSync}
          onRenameWorkflow={handleRenameWorkflow}
          onDeleteWorkflow={handleDeleteWorkflow}
          readOnly={isWorkflowReadOnly}
        />
        <ResizableHandle withHandle />
        <CanvasSection canvasProps={canvasProps}>
          <ContextualPane
            selectedNode={selectedNode}
            allNodes={allNodes}
            allEdges={allEdges}
            onUpdate={handleNodeConfigUpdateWithLock}
            workflowId={effectiveWorkflowId}
            workflowInputsDef={workflowInputsDef}
            triggers={workflowTriggers}
            readOnly={isWorkflowReadOnly}
          />
        </CanvasSection>
      </ResizablePanelGroup>
      <WorkflowDialogs
        isInputDialogOpen={isInputDialogOpen}
        isKeymapOpen={isKeymapOpen}
        inputFields={inputFields}
        inputData={inputData}
        setInputDialogOpen={setInputDialogOpen}
        setInputData={setInputData}
        setKeymapOpen={setKeymapOpen}
        onExecuteWithInput={() => handleExecuteWithInput(inputFields, inputData)}
        // Run options dialog
        isRunOptionsDialogOpen={isRunOptionsDialogOpen}
        setRunOptionsDialogOpen={setRunOptionsDialogOpen}
        triggersWithConnection={triggersWithConnection}
        onSelectTriggerForRealEvent={handleSelectTriggerForRealEvent}
        onConnectProvider={handleConnectProvider}
        onTriggerNow={handleTriggerNow}
        isTriggering={isTriggering}
        // Trigger event picker
        isTriggerEventPickerOpen={isTriggerEventPickerOpen}
        setTriggerEventPickerOpen={setTriggerEventPickerOpen}
        selectedTriggerForPicker={selectedTriggerForPicker}
        onTriggerEventSelect={handleTriggerEventSelect}
      />
    </motion.div>
  );
}
