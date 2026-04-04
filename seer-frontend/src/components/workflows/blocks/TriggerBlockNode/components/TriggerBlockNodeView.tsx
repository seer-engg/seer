import { useMemo, useState } from 'react';
import { Calendar } from 'lucide-react';
import { Handle } from '@xyflow/react';
import { cn } from '@/lib/utils';
import { useTemplateAutocomplete } from '@/hooks/useTemplateAutocomplete';
import type { WorkflowNodeData } from '../../../types';
import { OutputEdgeButton } from '../../../canvas/OutputEdgeButton';
import { NodeDeleteButton } from '../../../canvas/NodeDeleteButton';
import type { TriggerDescriptor } from '@/types/triggers';
import { TriggerHeader } from './TriggerHeader';
import { DynamicTriggerConfigForm } from '../triggers/DynamicTriggerConfigForm';
import { FormConfigEditor } from '../triggers/FormConfigEditor';
import { UserSchemaEditor } from '../triggers/UserSchemaEditor';
import type { BindingState } from '../../../triggers/utils';
import type { QuickOption } from './BaseTriggerNode';
import type { TriggerKind, TriggerConfigState } from '../triggers/useTriggerConfig';
import {
  buildSupabaseConfigFromProviderConfig,
  validateSupabaseConfig,
} from '../../../triggers/utils';
import { Button } from '@/components/ui/button';
import { useActiveWorkflowId } from '@/hooks/useActiveWorkflowId';
import { useConnectIntegration } from '@/hooks/useConnectIntegration';
import { useToolingData } from '@/hooks/useToolingData';
import { SlackChannelStatus } from '../triggers/SlackTriggerConfig';
import { useWorkflowSave } from '@/hooks/useWorkflowSave';
import { useSupabaseBinding } from '../../../dialogs/ResourcePicker/hooks/useSupabaseBinding';
import { SupabaseBindingDialog } from '../../../dialogs/ResourcePicker/SupabaseBindingDialog';
import { SupabaseConfigurationGrid } from './SupabaseConfigurationGrid';
import { useTriggersStore } from '@/stores/triggersStore';
import { useCanvasStore } from '@/stores/canvasStore';
import { getHandlePositionProps, useHandleOrientation } from '../../../canvas/hooks/useHandleOrientation';
import { useWorkflowCanvasContext } from '../../../canvas/workflow-canvas-context';

export interface TriggerBlockNodeViewProps {
  nodeId?: string;
  label: string;
  dragging?: boolean;
  selected?: boolean;
  descriptor?: NonNullable<WorkflowNodeData['triggerMeta']>['descriptor'] | null;
  subscription?: NonNullable<WorkflowNodeData['triggerMeta']>['subscription'];
  triggerKey: string;
  triggerKind: TriggerKind;
}

export function TriggerBlockNodeView(props: TriggerBlockNodeViewProps) {
  const {
    nodeId,
    label,
    dragging,
    selected,
    descriptor,
    subscription,
    triggerKey,
    triggerKind,
  } = props;

  const { readOnly } = useWorkflowCanvasContext();
  const orientation = useHandleOrientation();
  const outputHandlePosition = getHandlePositionProps(orientation, 'output');

  return (
    <div
      className={cn(
        'group relative min-w-[200px] max-w-[240px] rounded-xl border-2 bg-card p-4 shadow-sm transition-[border,box-shadow]',
        selected ? 'border-primary shadow-lg' : 'border-border',
      )}
    >
      {/* Delete button - visible on hover */}
      {!readOnly && nodeId && <NodeDeleteButton nodeId={nodeId} />}

      <TriggerHeader
        label={label}
        descriptor={descriptor}
        subscription={subscription}
        triggerKey={triggerKey}
        triggerKind={triggerKind}
      />


      {/* Output handle for trigger edges */}
      <Handle
        type="source"
        position={outputHandlePosition.position}
        id="trigger-output"
        style={{
          position: 'absolute',
          ...outputHandlePosition.style,
        }}
        className="!w-3 !h-3 !bg-seer !border-2 !border-background"
      />

      {/* Output edge button - positioned near right handle */}
      {!readOnly && !dragging && nodeId && (
        <OutputEdgeButton
          nodeId={nodeId}
          nodeType="trigger"
        />
      )}
    </div>
  );
}

/**
 * Shows cron details from dynamic config values.
 */
export function CronDetailsSectionFromConfig({ configValues }: { configValues: Record<string, unknown> }) {
  const cronExpression = String(configValues.cron_expression ?? '');
  const timezone = String(configValues.timezone ?? 'UTC');
  const description = configValues.description ? String(configValues.description) : undefined;

  if (!cronExpression) {
    return null;
  }

  return (
    <div className="rounded-md border border-dashed p-3 space-y-2 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Calendar className="h-4 w-4" />
        Schedule configuration
      </div>
      <div className="space-y-1.5">
        <p className="text-xs text-muted-foreground">Expression</p>
        <code className="text-xs block">{cronExpression}</code>
      </div>
      <div className="space-y-1.5">
        <p className="text-xs text-muted-foreground">Timezone</p>
        <p className="text-xs font-medium">{timezone}</p>
      </div>
      {description && (
        <div className="space-y-1.5 pt-1.5 border-t border-dashed border-border/60">
          <p className="text-xs text-muted-foreground">Description</p>
          <p className="text-xs">{description}</p>
        </div>
      )}
    </div>
  );
}

export interface ExpandedConfigSectionProps {
  triggerKind: TriggerKind;
  config: TriggerConfigState;
  descriptor?: TriggerDescriptor;
  subscription: NonNullable<WorkflowNodeData['triggerMeta']>['subscription'];
  templateAutocomplete: ReturnType<typeof useTemplateAutocomplete>;
  bindingState: BindingState;
  handleBindingModeChange: (inputName: string, mode: 'event' | 'literal') => void;
  handleBindingValueChange: (inputName: string, value: string) => void;
  bindingQuickInsert: (inputName: string, value: string) => void;
  quickOptions: QuickOption[];
}

export function ExpandedConfigSection({
  triggerKind,
  config,
  descriptor,
  subscription,
  templateAutocomplete,
  bindingState,
  handleBindingModeChange,
  handleBindingValueChange,
  bindingQuickInsert,
  quickOptions,
}: ExpandedConfigSectionProps) {
  const workflowId = useActiveWorkflowId();
  const markDirty = useCanvasStore((state) => state.markDirty);
  const isSupabaseTrigger = triggerKind === 'supabase';
  const supabaseConfig =
    isSupabaseTrigger && config.kind === 'dynamic'
      ? buildSupabaseConfigFromProviderConfig(config.configValues)
      : null;
  const supabaseValidation = supabaseConfig ? validateSupabaseConfig(supabaseConfig) : null;

  // Helper: Extract form config from ui_meta
  const extractFormConfig = () => {
    const uiMeta = subscription?.ui_meta as Record<string, unknown> | undefined;
    const formConfig = (uiMeta?.form_config as Record<string, string>) || {};
    return {
      title: formConfig.title || 'Form',
      description: formConfig.description || '',
      submitButtonText: formConfig.submitButtonText || 'Submit',
      successMessage: formConfig.successMessage || 'Thank you for your submission!',
    };
  };

  // Helper: Update form config in ui_meta
  const updateFormConfig = (formConfigValues: Record<string, string>) => {
    if (!workflowId || !subscription?.id) {
      return;
    }

    useTriggersStore.getState().updateTrigger(workflowId, subscription.id, {
      ui_meta: {
        ...(subscription.ui_meta as Record<string, unknown> || {}),
        form_config: formConfigValues,
      },
    });

    markDirty(); // Trigger autosave
  };

  return (
    <div className="space-y-4">
      {/* Dynamic config form for triggers with config_schema */}
      {config.kind === 'dynamic' && descriptor?.config_schema && !isSupabaseTrigger && (
        <DynamicTriggerConfigForm
          configSchema={descriptor.config_schema}
          configValues={config.configValues}
          onConfigChange={config.setConfigValue}
          provider={descriptor.provider}
          templateAutocomplete={templateAutocomplete}
        />
      )}

      {/* Slack channel bot membership status */}
      {triggerKind === 'slack' && config.kind === 'dynamic' && (
        <SlackChannelStatus
          workspaceId={config.configValues.workspace_id as string | undefined}
          channelId={config.configValues.channel_id as string | undefined}
        />
      )}

      {isSupabaseTrigger && config.kind === 'dynamic' && supabaseConfig && (
        <SupabaseConfigSection
          supabaseConfig={supabaseConfig}
          validation={supabaseValidation}
          setConfigValue={config.setConfigValue}
        />
      )}

      {/* User schema editor for form triggers only — webhook schema comes from received events */}
      {config.kind === 'webhook' && triggerKind === 'form' && (
        <>
          <FormConfigEditor
            initialConfig={extractFormConfig()}
            onChange={updateFormConfig}
          />

          <UserSchemaEditor
            schema={config.dataSchema}
            onChange={config.setDataSchema}
          />
        </>
      )}
    </div>
  );
}

function SupabaseConfigSection({
  supabaseConfig,
  validation,
  setConfigValue,
}: {
  supabaseConfig: SupabaseConfigState;
  validation: ReturnType<typeof validateSupabaseConfig> | null;
  setConfigValue: (key: string, value: unknown) => void;
}) {
  const [isConnecting, setIsConnecting] = useState(false);
  const connectIntegration = useConnectIntegration();
  const { isIntegrationConnected } = useToolingData();
  const supabaseConnected = isIntegrationConnected('supabase');
  const { saveWorkflow, hasWorkflow } = useWorkflowSave();
  const bindingState = useSupabaseBinding();

  const handleConnectSupabase = async () => {
    try {
      setIsConnecting(true);
      if (hasWorkflow) {
        await saveWorkflow();
      }
      const redirectUrl = await connectIntegration('supabase', { toolNames: ['supabase_table_query'] });
      if (redirectUrl) {
        window.location.href = redirectUrl;
      }
    } catch (error) {
      console.error('Failed to connect Supabase', error);
    } finally {
      setIsConnecting(false);
    }
  };

  const firstError = useMemo(() => {
    if (!validation || validation.valid) return null;
    return Object.values(validation.errors).find(Boolean) || null;
  }, [validation]);

  return (
    <div className="space-y-4">
      <SupabaseConfigurationGrid
        supabaseConfig={supabaseConfig}
        validation={validation}
        setConfigValue={setConfigValue}
        supabaseConnected={supabaseConnected}
        isConnecting={isConnecting}
        onConnect={handleConnectSupabase}
        onOpenBinding={() => bindingState.setBindModalOpen(true)}
        bindingState={bindingState}
      />

      {firstError && (
        <p className="text-xs text-destructive">
          {firstError}
        </p>
      )}

      <SupabaseBindingDialog
        open={bindingState.bindModalOpen}
        onOpenChange={bindingState.setBindModalOpen}
        bindingState={bindingState}
        onBindSuccess={(resource, fallback) => {
          setConfigValue('integration_resource_id', resource.id);
          setConfigValue('integration_resource_label', resource.name || fallback?.displayName || fallback?.projectRef);
          setConfigValue('schema', 'public');
          setConfigValue('table', '');
        }}
        refetchResources={() => bindingState.fetchNextSupabaseProjects?.()}
      />
    </div>
  );
}
