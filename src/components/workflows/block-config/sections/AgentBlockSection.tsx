import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { backendApiClient } from '@/lib/api-client';
import { memoryApi } from '@/lib/memory-api';
import { memoryKeys } from '@/lib/query-keys';
import type { ModelDescriptor } from '@/components/workflows/buildtypes';

import { StructuredOutputEditor } from '@/components/workflows/editors/StructuredOutputEditor';
import { AgentBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';
import { AgentToolsSelector } from '../widgets/AgentToolsSelector';

interface AgentBlockSectionProps extends BlockSectionProps<AgentBlockConfig> {
  useStructuredOutput: boolean;
  setUseStructuredOutput: (value: boolean) => void;
  structuredOutputSchema?: Record<string, unknown>;
  onStructuredOutputSchemaChange: (schema?: Record<string, unknown>) => void;
}

const DEFAULT_MODELS = [
  { value: 'qwen/qwen3-235b-a22b-2507', label: 'Qwen 3' },
  { value: 'google/gemini-2.0-flash-001', label: 'Gemini' },
  { value: 'mistralai/mistral-small-3.2-24b-instruct', label: 'Mistral' },
  { value: 'moonshotai/kimi-k2.5', label: 'Kimi' },
];

function useModelOptions() {
  const [modelOptions, setModelOptions] = useState<{ value: string; label: string }[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        setModelsLoading(true);
        const response = await backendApiClient.request<{ models: ModelDescriptor[] }>('/api/v1/registries/models');
        setModelOptions(response.models.map((m) => ({ value: m.id, label: m.title })));
      } catch {
        setModelOptions(DEFAULT_MODELS);
      } finally {
        setModelsLoading(false);
      }
    })();
  }, []);

  return { modelOptions, modelsLoading };
}

function useMemoryBankFieldData(selectedMemoryBankId: string | null | undefined) {
  const memoryBanksQuery = useQuery({
    queryKey: memoryKeys.banks(),
    queryFn: memoryApi.listBanks,
    retry: false,
  });
  const memoryBanks = useMemo(() => memoryBanksQuery.data?.items ?? [], [memoryBanksQuery.data?.items]);
  const memoryBankOptions = useMemo(() => {
    const options = ['__none__', ...memoryBanks.map((bank) => bank.memory_bank_id)];
    if (selectedMemoryBankId && !options.includes(selectedMemoryBankId)) {
      options.push(selectedMemoryBankId);
    }
    return options;
  }, [memoryBanks, selectedMemoryBankId]);
  const memoryBankLabels = useMemo(() => {
    const labels = ['No memory bank', ...memoryBanks.map((bank) => bank.name)];
    if (selectedMemoryBankId && !memoryBanks.some((bank) => bank.memory_bank_id === selectedMemoryBankId)) {
      labels.push(selectedMemoryBankId);
    }
    return labels;
  }, [memoryBanks, selectedMemoryBankId]);
  const description = memoryBanksQuery.isLoading
    ? 'Loading memory banks...'
    : memoryBanksQuery.isError
      ? 'Memory banks are unavailable right now.'
      : 'Optionally attach a workspace memory bank to this agent.';

  return { description, memoryBankLabels, memoryBankOptions };
}

function useResolvedModelOptions(configModel: string | undefined, setConfig: AgentBlockSectionProps['setConfig']) {
  const { modelOptions, modelsLoading } = useModelOptions();
  const effectiveModelOptions = useMemo(() => {
    if (configModel && !modelOptions.find((model) => model.value === configModel)) {
      return [...modelOptions, { value: configModel, label: configModel }];
    }
    return modelOptions;
  }, [configModel, modelOptions]);

  useEffect(() => {
    if (!configModel && modelOptions.length > 0) {
      const defaultModel = modelOptions.find((model) => model.value === 'qwen/qwen3-235b-a22b-2507');
      setConfig((prev) => ({ ...prev, model: defaultModel?.value || modelOptions[0].value }));
    }
  }, [configModel, modelOptions, setConfig]);

  return { effectiveModelOptions, modelOptions, modelsLoading };
}

function StructuredOutputToggle({
  useStructuredOutput,
  setUseStructuredOutput,
  onStructuredOutputSchemaChange,
  outputSchema,
}: {
  useStructuredOutput: boolean;
  setUseStructuredOutput: (value: boolean) => void;
  onStructuredOutputSchemaChange: (schema?: Record<string, unknown>) => void;
  outputSchema?: Record<string, unknown>;
}) {
  return (
    <>
      <div className="flex items-center space-x-2 pt-2 border-t">
        <Checkbox
          id="agent-structured-output"
          checked={useStructuredOutput}
          onCheckedChange={checked => {
            const isChecked = checked === true;
            setUseStructuredOutput(isChecked);
            if (!isChecked) {
              onStructuredOutputSchemaChange(undefined);
            } else if (!outputSchema) {
              onStructuredOutputSchemaChange({
                type: 'object',
                properties: {},
              });
            }
          }}
        />
        <Label htmlFor="agent-structured-output" className="text-sm font-normal cursor-pointer">
          Structured Output (JSON Schema)
        </Label>
      </div>

      {useStructuredOutput && (
        <StructuredOutputEditor
          value={outputSchema}
          onChange={schema => onStructuredOutputSchemaChange(schema)}
        />
      )}
    </>
  );
}

function AgentPromptField({
  config,
  setConfig,
  templateAutocomplete,
  error,
}: Pick<AgentBlockSectionProps, 'config' | 'setConfig' | 'templateAutocomplete'> & {
  error?: string;
}) {
  return (
    <DynamicFormField
      name="prompt"
      label="Agent Instructions"
      description="Describe what you want the agent to accomplish"
      required
      value={config.prompt || ''}
      onChange={(value) => setConfig((prev) => ({ ...prev, prompt: value as string }))}
      def={{ type: 'string', multiline: true }}
      templateAutocomplete={templateAutocomplete}
      rows={12}
      error={error}
    />
  );
}

function AgentModelField({
  config,
  setConfig,
  templateAutocomplete,
  error,
  modelOptions,
  effectiveModelOptions,
  modelsLoading,
}: Pick<AgentBlockSectionProps, 'config' | 'setConfig' | 'templateAutocomplete'> & {
  error?: string;
  modelOptions: { value: string; label: string }[];
  effectiveModelOptions: { value: string; label: string }[];
  modelsLoading: boolean;
}) {
  return (
    <DynamicFormField
      name="model"
      label="Model"
      description={modelsLoading ? 'Loading models...' : 'Choose the AI model for the agent'}
      defaultValue={modelOptions[0]?.label}
      value={config.model || modelOptions[0]?.value || 'qwen/qwen3-235b-a22b-2507'}
      onChange={(value) => setConfig((prev) => ({ ...prev, model: String(value) }))}
      def={{ type: 'string', enum: effectiveModelOptions.map((option) => option.value) }}
      templateAutocomplete={templateAutocomplete}
      error={error}
    />
  );
}

function AgentMemoryBankField({
  config,
  setConfig,
  templateAutocomplete,
  error,
  description,
  memoryBankOptions,
  memoryBankLabels,
}: Pick<AgentBlockSectionProps, 'config' | 'setConfig' | 'templateAutocomplete'> & {
  error?: string;
  description: string;
  memoryBankOptions: string[];
  memoryBankLabels: string[];
}) {
  return (
    <DynamicFormField
      name="memory_bank_id"
      label="Memory Bank"
      description={description}
      value={config.memory_bank_id ?? '__none__'}
      onChange={(value) =>
        setConfig((prev) => ({
          ...prev,
          memory_bank_id: value === '__none__' ? null : String(value),
        }))
      }
      def={{
        type: 'string',
        enum: memoryBankOptions,
        enumLabels: memoryBankLabels,
      }}
      templateAutocomplete={templateAutocomplete}
      error={error}
    />
  );
}

function EnableArtifactsToggle({
  config,
  setConfig,
}: Pick<AgentBlockSectionProps, 'config' | 'setConfig'>) {
  return (
    <>
      <div className="flex items-center space-x-2 pt-2 border-t">
        <Checkbox
          id="agent-enable-artifacts"
          checked={config.enable_artifacts === true}
          onCheckedChange={(checked) =>
            setConfig((prev) => ({ ...prev, enable_artifacts: checked === true }))
          }
        />
        <Label htmlFor="agent-enable-artifacts" className="text-sm font-normal cursor-pointer">
          Enable Artifacts
        </Label>
      </div>
      <p className="text-xs text-muted-foreground -mt-2 ml-6">
        Allow the agent to generate downloadable PDF and DOCX files.
      </p>
    </>
  );
}

function MaxIterationsField({
  config,
  setConfig,
  templateAutocomplete,
  error,
}: Pick<AgentBlockSectionProps, 'config' | 'setConfig' | 'templateAutocomplete'> & {
  error?: string;
}) {
  return (
    <div className="space-y-1.5">
      <Label className="text-sm font-medium">Max Iterations</Label>
      <DynamicFormField
        name="max_iterations"
        description="Maximum number of reasoning and tool-use iterations (1-100)"
        defaultValue={10}
        value={config.max_iterations ?? 10}
        onChange={(value) =>
          setConfig((prev) => ({
            ...prev,
            max_iterations: typeof value === 'string' ? parseInt(value, 10) : (value as number),
          }))
        }
        def={{ type: 'integer', minimum: 1, maximum: 100 }}
        templateAutocomplete={templateAutocomplete}
        error={error}
      />
    </div>
  );
}

export function AgentBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
  useStructuredOutput,
  setUseStructuredOutput,
  structuredOutputSchema,
  onStructuredOutputSchemaChange,
}: AgentBlockSectionProps) {
  const outputSchema = structuredOutputSchema ?? config.output_schema;
  const { effectiveModelOptions, modelOptions, modelsLoading } = useResolvedModelOptions(
    config.model,
    setConfig
  );
  const { description, memoryBankLabels, memoryBankOptions } = useMemoryBankFieldData(
    config.memory_bank_id
  );

  const handleToolsChange = useCallback(
    (tools: string[]) => {
      setConfig((prev) => ({ ...prev, tools }));
    },
    [setConfig]
  );

  const handleConnectionChange = useCallback(
    (toolName: string, connectionId: number | null) => {
      setConfig((prev) => ({
        ...prev,
        tool_connections: {
          ...prev.tool_connections,
          [toolName]: connectionId,
        },
      }));
    },
    [setConfig]
  );

  const handleResourceIdChange = useCallback(
    (toolName: string, resourceId: number | null) => {
      setConfig((prev) => ({
        ...prev,
        tool_resource_ids: {
          ...prev.tool_resource_ids,
          [toolName]: resourceId,
        },
      }));
    },
    [setConfig]
  );

  return (
    <div className="space-y-4">
      <AgentPromptField
        config={config}
        setConfig={setConfig}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['prompt']}
      />

      <AgentModelField
        config={config}
        setConfig={setConfig}
        templateAutocomplete={templateAutocomplete}
        modelOptions={modelOptions}
        effectiveModelOptions={effectiveModelOptions}
        modelsLoading={modelsLoading}
        error={validationErrors['model']}
      />

      <AgentMemoryBankField
        config={config}
        setConfig={setConfig}
        templateAutocomplete={templateAutocomplete}
        description={description}
        memoryBankOptions={memoryBankOptions}
        memoryBankLabels={memoryBankLabels}
        error={validationErrors['memory_bank_id']}
      />

      <AgentToolsSelector
        selectedTools={config.tools || []}
        toolConnections={config.tool_connections || {}}
        toolResourceIds={config.tool_resource_ids || {}}
        onToolsChange={handleToolsChange}
        onConnectionChange={handleConnectionChange}
        onResourceIdChange={handleResourceIdChange}
      />

      <MaxIterationsField
        config={config}
        setConfig={setConfig}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['max_iterations']}
      />

      <EnableArtifactsToggle config={config} setConfig={setConfig} />

      <StructuredOutputToggle
        useStructuredOutput={useStructuredOutput}
        setUseStructuredOutput={setUseStructuredOutput}
        onStructuredOutputSchemaChange={onStructuredOutputSchemaChange}
        outputSchema={outputSchema}
      />
    </div>
  );
}
