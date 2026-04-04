import { useEffect, useCallback, useMemo, useState } from 'react';
import { Node } from '@xyflow/react';

import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Check, ChevronsUpDown } from 'lucide-react';
import { cn } from '@/lib/utils';

import {
  ToolBlockConfig,
  type ResourcePickerConfig,
  ToolMetadata,
  BlockSectionProps,
  TemplateAutocompleteControls,
} from '../types';
import type { WorkflowEdge, WorkflowNodeData } from '@/components/workflows/types';
import { DynamicFormField } from '../widgets/DynamicFormField';
import type { ToolParamDefinition } from '../widgets/ParamInputFactory';
import { AccountSelector } from '../widgets/AccountSelector';
import { useConnectIntegration } from '@/hooks/useConnectIntegration';
import { useToolingData } from '@/hooks/useToolingData';
import { discoverFileVariables, type FileVariable } from '../helpers/discoverFileVariables';
import { SlackChannelStatus } from '@/components/workflows/blocks/TriggerBlockNode/triggers/SlackTriggerConfig';

interface ToolBlockSectionProps extends BlockSectionProps<ToolBlockConfig> {
  toolSchema?: ToolMetadata;
  /** All nodes in workflow for file variable discovery */
  allNodes?: Node<WorkflowNodeData>[];
  /** All edges in workflow for file variable discovery */
  allEdges?: WorkflowEdge[];
  /** Current node being configured */
  currentNode?: Node<WorkflowNodeData>;
}

function JsonParamsEditor({
  toolParams,
  setConfig,
}: {
  toolParams: Record<string, unknown>;
  setConfig: (updater: (prev: ToolBlockConfig) => ToolBlockConfig) => void;
}) {
  return (
    <div>
      <Label htmlFor="tool-params">Parameters (JSON)</Label>
      <Textarea
        id="tool-params"
        value={JSON.stringify(toolParams, null, 2)}
        onChange={e => {
          try {
            const params = JSON.parse(e.target.value);
            setConfig(prev => ({ ...prev, params }));
          } catch {
            // Ignore invalid JSON
          }
        }}
        placeholder='{"max_results": 5}'
        className="font-mono text-xs"
        rows={4}
      />
    </div>
  );
}

function updateConfigResourceLabels(
  prev: ToolBlockConfig,
  paramName: string,
  label?: string,
): ToolBlockConfig {
  const currentLabels = (prev.__resourceLabels as Record<string, string> | undefined) || {};
  const nextLabels = { ...currentLabels };

  if (label) {
    nextLabels[paramName] = label;
  } else {
    delete nextLabels[paramName];
  }

  if (Object.keys(nextLabels).length === 0) {
    const { __resourceLabels, ...rest } = prev;
    return rest;
  }

  return {
    ...prev,
    __resourceLabels: nextLabels,
  };
}

function syncOutputSchema(
  config: ToolBlockConfig,
  toolSchema?: ToolMetadata,
): ToolBlockConfig {
  if (!toolSchema?.output_schema || config.output_schema) {
    return config;
  }
  return {
    ...config,
    output_schema: toolSchema.output_schema,
  };
}

function useToolConfigSync({
  integrationType,
  toolName,
  toolsInGroup,
  toolSchema,
  setConfig,
}: {
  integrationType?: string | null;
  toolName?: string | null;
  toolsInGroup: ToolMetadata[];
  toolSchema?: ToolMetadata;
  setConfig: (updater: (prev: ToolBlockConfig) => ToolBlockConfig) => void;
}) {
  useEffect(() => {
    setConfig(prev => syncOutputSchema(prev, toolSchema));
  }, [toolSchema, setConfig]);

  useEffect(() => {
    if (integrationType && !toolName && toolsInGroup.length > 0) {
      const firstTool = toolsInGroup[0];
      setConfig(prev => ({
        ...prev,
        tool_name: firstTool.name,
        output_schema: firstTool.output_schema || undefined,
        params: {},
      }));
    }
  }, [integrationType, toolName, toolsInGroup, setConfig]);

  useEffect(() => {
    if (toolName && !integrationType && toolSchema?.integration_type) {
      setConfig(prev => ({
        ...prev,
        integration_type: toolSchema.integration_type!,
      }));
    }
  }, [toolName, integrationType, toolSchema, setConfig]);
}

interface ParamFieldProps {
  paramName: string;
  paramDef: ToolParamDefinition;
  toolParams: Record<string, unknown>;
  requiredParams: string[];
  toolProvider: string;
  updateParams: (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => void;
  updateResourceLabel: (paramName: string, label?: string) => void;
  templateAutocomplete: TemplateAutocompleteControls;
  error?: string;
  /** Available file variables from upstream nodes */
  availableFileVariables?: FileVariable[];
}

function ParamField({
  paramName,
  paramDef,
  toolParams,
  requiredParams,
  toolProvider,
  updateParams,
  updateResourceLabel,
  templateAutocomplete,
  error,
  availableFileVariables = [],
}: ParamFieldProps) {
  const paramValue = toolParams[paramName];
  const isRequired = requiredParams.includes(paramName);
  const resourcePicker = paramDef['x-resource-picker'] as ResourcePickerConfig | undefined;
  const dependsOnKey = resourcePicker?.depends_on;
  const dependsOnValues = dependsOnKey ? { [dependsOnKey]: toolParams[dependsOnKey] as string } : undefined;

  return (
    <DynamicFormField
      name={paramName}
      label={paramName}
      description={paramDef.description as string | undefined}
      required={isRequired}
      defaultValue={paramDef.default}
      value={paramValue}
      onChange={val => {
        updateParams(prev => ({
          ...prev,
          [paramName]: val,
        }));
      }}
      def={paramDef}
      provider={toolProvider}
      dependsOnValues={dependsOnValues}
      templateAutocomplete={templateAutocomplete}
      error={error}
      onResourceLabelChange={(field, label) => updateResourceLabel(field, label)}
      availableFileVariables={availableFileVariables}
    />
  );
}

function ToolSelector({
  integrationType,
  toolsInGroup,
  toolName,
  onChange,
}: {
  integrationType?: string | null;
  toolsInGroup: ToolMetadata[];
  toolName?: string | null;
  onChange: (toolName: string) => void;
}) {
  const [open, setOpen] = useState(false);

  if (!integrationType || toolsInGroup.length === 0) {
    return null;
  }

  return (
    <div className="space-y-2">
      <Label htmlFor="tool-select">Tool</Label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            id="tool-select"
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            {toolName ? toolsInGroup.find((tool) => tool.name === toolName)?.name : 'Select a tool...'}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search tools..." />
            <CommandList>
              <CommandEmpty>No tool found.</CommandEmpty>
              <CommandGroup>
                {toolsInGroup.map((tool) => (
                  <CommandItem
                    key={tool.name}
                    value={tool.name}
                    onSelect={(currentValue) => {
                      onChange(currentValue);
                      setOpen(false);
                    }}
                    className="flex flex-col items-start gap-1 py-2"
                  >
                    <div className="flex items-center gap-2 w-full">
                      <Check
                        className={cn(
                          'h-4 w-4 flex-shrink-0',
                          toolName === tool.name ? 'opacity-100' : 'opacity-0',
                        )}
                      />
                      <span className="font-medium">{tool.name}</span>
                    </div>
                    {tool.description && (
                      <span className="text-xs text-muted-foreground ml-6 line-clamp-2">
                        {tool.description}
                      </span>
                    )}
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}

/**
 * Renders Slack channel bot membership status for Slack tools.
 * Only shown when workspace_id and channel_id are both selected.
 */
function SlackToolChannelStatus({ toolParams, integrationType }: {
  toolParams: Record<string, unknown>;
  integrationType?: string | null;
}) {
  if (integrationType !== 'slack') return null;

  const workspaceId = toolParams.workspace_id as string | undefined;
  const channelId = toolParams.channel_id as string | undefined;

  if (!workspaceId || !channelId) return null;

  return <SlackChannelStatus workspaceId={workspaceId} channelId={channelId} />;
}

interface ToolParametersProps {
  integrationType?: string | null;
  toolName?: string | null;
  hasParamSchema: boolean;
  paramSchema: Record<string, ToolParamDefinition>;
  toolParams: Record<string, unknown>;
  requiredParams: string[];
  toolProvider: string;
  updateParams: (updater: (prev: Record<string, unknown>) => Record<string, unknown>) => void;
  updateResourceLabel: (paramName: string, label?: string) => void;
  templateAutocomplete: TemplateAutocompleteControls;
  validationErrors: Record<string, string>;
  setConfig: (updater: (prev: ToolBlockConfig) => ToolBlockConfig) => void;
  availableFileVariables?: FileVariable[];
}

function ToolParameters({ integrationType, toolName, hasParamSchema, paramSchema, toolParams, requiredParams, toolProvider, updateParams, updateResourceLabel, templateAutocomplete, validationErrors, setConfig, availableFileVariables = [] }: ToolParametersProps) {
  if (!toolName && integrationType) return <div className="text-sm text-muted-foreground py-4 text-center">Select a tool above to configure parameters</div>;
  if (!toolName) return null;
  if (!hasParamSchema) return <JsonParamsEditor toolParams={toolParams} setConfig={setConfig} />;

  // Filter out hidden fields (x-ui-hidden: true)
  const visibleParams = Object.entries(paramSchema).filter(
    ([, def]) => def['x-ui-hidden'] !== true
  );

  return (
    <div className="space-y-2">
      {visibleParams.map(([name, def]) => (
        <ParamField key={name} paramName={name} paramDef={def} toolParams={toolParams} requiredParams={requiredParams} toolProvider={toolProvider} updateParams={updateParams} updateResourceLabel={updateResourceLabel} templateAutocomplete={templateAutocomplete} error={validationErrors?.[name]} availableFileVariables={availableFileVariables} />
      ))}
    </div>
  );
}

function useToolAccountConfig(toolSchema: ToolMetadata | undefined, integrationType: string | undefined, toolName: string | undefined, setConfig: (updater: (prev: ToolBlockConfig) => ToolBlockConfig) => void) {
  const connectIntegration = useConnectIntegration();
  const toolRequiresOAuth = Boolean(toolSchema?.provider || toolSchema?.integration_type);

  const handleAccountSelect = useCallback((connectionId: number | null) => {
    setConfig(prev => ({ ...prev, connection_id: connectionId }));
  }, [setConfig]);

  const handleConnectAccount = useCallback(async () => {
    const type = toolSchema?.integration_type || integrationType;
    if (!type || !toolName) return;
    try {
      const redirectUrl = await connectIntegration(type as Parameters<typeof connectIntegration>[0], { toolName, forceNewAccount: true });
      if (redirectUrl) window.location.href = redirectUrl;
    } catch (error) {
      console.error('Failed to initiate OAuth connection:', error);
    }
  }, [toolSchema, integrationType, toolName, connectIntegration]);

  return { toolRequiresOAuth, handleAccountSelect, handleConnectAccount };
}

function useToolParamsConfig(config: ToolBlockConfig, toolSchema: ToolMetadata | undefined, toolsInGroup: ToolMetadata[], setConfig: (updater: (prev: ToolBlockConfig) => ToolBlockConfig) => void) {
  const toolParams = config.params ?? {};
  const paramSchema = (toolSchema?.parameters?.properties ?? {}) as Record<string, ToolParamDefinition>;
  const requiredParams = toolSchema?.parameters?.required ?? [];
  const toolProvider = config.provider ?? toolSchema?.provider ?? 'google';
  const hasParamSchema = Object.keys(paramSchema).length > 0;

  const updateResourceLabel = useCallback((paramName: string, label?: string) => { setConfig(prev => updateConfigResourceLabels(prev, paramName, label)); }, [setConfig]);
  const updateParams = useCallback((updater: (prev: Record<string, unknown>) => Record<string, unknown>) => { setConfig(prev => ({ ...prev, params: updater(prev.params ?? {}) })); }, [setConfig]);
  const handleToolChange = useCallback((newToolName: string) => {
    const newTool = toolsInGroup.find((t) => t.name === newToolName);
    if (newTool) setConfig(prev => ({ ...prev, tool_name: newToolName, output_schema: newTool.output_schema ?? undefined, params: {} }));
  }, [toolsInGroup, setConfig]);

  return { toolParams, paramSchema, requiredParams, toolProvider, hasParamSchema, updateResourceLabel, updateParams, handleToolChange };
}

export function ToolBlockSection({ config, setConfig, toolSchema, templateAutocomplete, validationErrors = {}, allNodes = [], allEdges = [], currentNode }: ToolBlockSectionProps) {
  const integrationType = config.integration_type;
  const toolName = config.tool_name ?? config.toolName;
  const availableFileVariables = useMemo(() => discoverFileVariables(allNodes, allEdges, currentNode ?? null), [allNodes, allEdges, currentNode]);
  const { getToolsByIntegration } = useToolingData();
  const toolsInGroup = useMemo(() => (integrationType ? getToolsByIntegration(integrationType) : []), [integrationType, getToolsByIntegration]);

  const { toolRequiresOAuth, handleAccountSelect, handleConnectAccount } = useToolAccountConfig(toolSchema, integrationType, toolName, setConfig);
  const { toolParams, paramSchema, requiredParams, toolProvider, hasParamSchema, updateResourceLabel, updateParams, handleToolChange } = useToolParamsConfig(config, toolSchema, toolsInGroup, setConfig);

  useToolConfigSync({ integrationType, toolName, toolsInGroup, toolSchema, setConfig });

  if (!integrationType && !hasParamSchema) return <JsonParamsEditor toolParams={toolParams} setConfig={setConfig} />;

  return (
    <div className="space-y-4">
      <ToolSelector integrationType={integrationType} toolsInGroup={toolsInGroup} toolName={toolName} onChange={handleToolChange} />
      {toolName && toolRequiresOAuth && (
        <AccountSelector toolName={toolName} selectedConnectionId={config.connection_id} onSelect={handleAccountSelect} error={validationErrors?.connection_id} onConnectAccount={handleConnectAccount} />
      )}
      <ToolParameters integrationType={integrationType} toolName={toolName} hasParamSchema={hasParamSchema} paramSchema={paramSchema} toolParams={toolParams} requiredParams={requiredParams} toolProvider={toolProvider} updateParams={updateParams} updateResourceLabel={updateResourceLabel} templateAutocomplete={templateAutocomplete} validationErrors={validationErrors} setConfig={setConfig} availableFileVariables={availableFileVariables} />
      <SlackToolChannelStatus toolParams={toolParams} integrationType={integrationType} />
    </div>
  );
}
