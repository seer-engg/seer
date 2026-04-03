import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, ChevronDown, Globe, Loader2 } from 'lucide-react';
import { backendApiClient } from '@/lib/api-client';
import type { ModelDescriptor } from '@/components/workflows/buildtypes';

import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

import { StructuredOutputEditor } from '@/components/workflows/editors/StructuredOutputEditor';
import { BrowserBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';
import { useBrowserProfiles } from '@/hooks/useBrowserProfiles';

interface BrowserBlockSectionProps extends BlockSectionProps<BrowserBlockConfig> {
  useStructuredOutput: boolean;
  setUseStructuredOutput: (value: boolean) => void;
  structuredOutputSchema?: Record<string, unknown>;
  onStructuredOutputSchemaChange: (schema?: Record<string, unknown>) => void;
}

type JsonSchema = { type: string; properties: Record<string, { type: string; description?: string }> };

interface BrowserProfile {
  id: string;
  name: string;
  logged_in_domains: string[];
}

function BrowserProfileSelector({
  value,
  onChange,
  profiles,
  isLoading,
  error,
}: {
  value: string | null | undefined;
  onChange: (value: string | null) => void;
  profiles: BrowserProfile[];
  isLoading: boolean;
  error?: string;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor="browser-profile" className="text-sm font-medium">Browser Profile (Optional)</Label>
      <Select
        value={value || '__none__'}
        onValueChange={(v) => onChange(v === '__none__' ? null : v)}
      >
        <SelectTrigger id="browser-profile" className="w-full">
          <SelectValue placeholder="Anonymous (no saved sessions)" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__none__">
            <span className="flex items-center gap-2">
              <Globe className="h-3.5 w-3.5 text-muted-foreground" />
              Anonymous (no saved sessions)
            </span>
          </SelectItem>
          {isLoading ? (
            <SelectItem value="__loading__" disabled>
              <span className="flex items-center gap-2">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Loading profiles...
              </span>
            </SelectItem>
          ) : (
            profiles.map((profile) => (
              <SelectItem key={profile.id} value={profile.id}>
                <span className="flex items-center gap-2">
                  {profile.name}
                  {profile.logged_in_domains.length > 0 && (
                    <span className="text-xs text-muted-foreground">
                      ({profile.logged_in_domains.length} site{profile.logged_in_domains.length !== 1 ? 's' : ''})
                    </span>
                  )}
                </span>
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">
          Select a profile with saved login sessions.
        </p>
        <Button variant="outline" size="sm" className="h-7 text-xs shrink-0" asChild>
          <Link to="/settings?tab=browser">
            Manage Profiles
            <ArrowUpRight className="h-3 w-3 ml-1" />
          </Link>
        </Button>
      </div>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
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
          id="browser-structured-output"
          checked={useStructuredOutput}
          onCheckedChange={checked => {
            const isChecked = checked === true;
            setUseStructuredOutput(isChecked);
            if (!isChecked) {
              onStructuredOutputSchemaChange(undefined);
            } else if (!outputSchema) {
              onStructuredOutputSchemaChange({ type: 'object', properties: {} });
            }
          }}
        />
        <Label htmlFor="browser-structured-output" className="text-sm font-normal cursor-pointer">
          Structured Output (JSON Schema)
        </Label>
      </div>

      {useStructuredOutput && (
        <>
          <p className="text-xs text-muted-foreground mb-2">
            Custom fields will be available under <code className="bg-muted px-1 rounded">extracted_data</code>.
            Standard outputs (<code className="bg-muted px-1 rounded">success</code>, <code className="bg-muted px-1 rounded">result</code>, <code className="bg-muted px-1 rounded">final_url</code>, <code className="bg-muted px-1 rounded">screenshots</code>) are always available.
          </p>
          <StructuredOutputEditor
            value={outputSchema as JsonSchema}
            onChange={schema => onStructuredOutputSchemaChange(schema as unknown as Record<string, unknown>)}
          />
        </>
      )}
    </>
  );
}

const DEFAULT_BROWSER_MODELS = [
  { value: 'qwen/qwen3-vl-8b-thinking', label: 'Qwen3 VL 8B' },
  { value: 'google/gemini-3.1-flash-lite-preview', label: 'Gemini 3.1 Flash Lite' },
];

function useBrowserModelOptions() {
  const [modelOptions, setModelOptions] = useState<{ value: string; label: string }[]>(DEFAULT_BROWSER_MODELS);

  useEffect(() => {
    (async () => {
      try {
        const response = await backendApiClient.request<{ models: ModelDescriptor[] }>('/api/v1/registries/browser-models');
        setModelOptions(response.models.map((m) => ({ value: m.id, label: m.title })));
      } catch {
        setModelOptions(DEFAULT_BROWSER_MODELS);
      }
    })();
  }, []);

  return modelOptions;
}

export function BrowserBlockSection({
  config, setConfig, templateAutocomplete, validationErrors = {},
  useStructuredOutput, setUseStructuredOutput, structuredOutputSchema, onStructuredOutputSchemaChange,
}: BrowserBlockSectionProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const browserModels = useBrowserModelOptions();
  const outputSchema = structuredOutputSchema ?? config.output_schema;
  const { profiles, isLoading: profilesLoading } = useBrowserProfiles();

  return (
    <div className="space-y-4">
      <DynamicFormField
        name="task" label="Task"
        description="Describe what you want the browser to do in natural language. Use ${variable} to reference upstream outputs."
        value={config.task || ''} rows={10}
        onChange={(value) => setConfig((prev) => ({ ...prev, task: String(value) }))}
        placeholder="e.g., Go to example.com and click the login button"
        def={{ type: 'string', multiline: true }} templateAutocomplete={templateAutocomplete} error={validationErrors['task']}
      />

      <DynamicFormField
        name="inputs" label="Context"
        description="Additional data to pass to the browser agent. Use ${variable} to reference upstream outputs."
        value={config.inputs || {}} rows={3}
        onChange={(value) => setConfig((prev) => ({ ...prev, inputs: value as Record<string, unknown> }))}
        def={{ type: 'object' }} templateAutocomplete={templateAutocomplete} error={validationErrors['inputs']}
      />

      <DynamicFormField
        name="model" label="Model"
        description="Vision model used for browser automation."
        value={config.model || browserModels[0]?.value || 'qwen/qwen3-vl-8b-thinking'}
        onChange={(value) => setConfig((prev) => ({ ...prev, model: String(value) }))}
        def={{ type: 'string', enum: browserModels.map((m) => m.value) }}
        templateAutocomplete={templateAutocomplete} error={validationErrors['model']}
      />

      <BrowserProfileSelector
        value={config.browser_profile_id}
        onChange={(value) => setConfig((prev) => ({ ...prev, browser_profile_id: value }))}
        profiles={profiles}
        isLoading={profilesLoading}
        error={validationErrors['browser_profile_id']}
      />

      <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
        <CollapsibleTrigger asChild>
          <Button variant="ghost" size="sm" className="w-full justify-between text-xs">
            Advanced Settings
            <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
          </Button>
        </CollapsibleTrigger>
        <CollapsibleContent className="space-y-4 pt-2">
          <DynamicFormField
            name="max_steps" label="Max Steps" description="Maximum number of browser actions (1-100). Default: 25"
            value={config.max_steps ?? 25} def={{ type: 'integer', minimum: 1, maximum: 100 }}
            onChange={(value) => setConfig((prev) => ({ ...prev, max_steps: Number(value) }))}
            templateAutocomplete={templateAutocomplete} error={validationErrors['max_steps']}
          />
          <DynamicFormField
            name="timeout_seconds" label="Timeout (seconds)" description="Maximum time for task completion (30-1800). Default: 300"
            value={config.timeout_seconds ?? 300} def={{ type: 'integer', minimum: 30, maximum: 1800 }}
            onChange={(value) => setConfig((prev) => ({ ...prev, timeout_seconds: Number(value) }))}
            templateAutocomplete={templateAutocomplete} error={validationErrors['timeout_seconds']}
          />
          <div className="flex items-center space-x-2 pt-2 border-t">
            <Checkbox id="save-screenshots" checked={config.save_screenshots === true}
              onCheckedChange={(checked) => setConfig((prev) => ({ ...prev, save_screenshots: checked === true }))} />
            <Label htmlFor="save-screenshots" className="text-sm font-normal cursor-pointer">Save Screenshots</Label>
          </div>
          <p className="text-xs text-muted-foreground -mt-2 ml-6">
            Save browser screenshots during execution. Screenshots will be available in the output.
          </p>
        </CollapsibleContent>
      </Collapsible>

      <StructuredOutputToggle
        useStructuredOutput={useStructuredOutput} setUseStructuredOutput={setUseStructuredOutput}
        onStructuredOutputSchemaChange={onStructuredOutputSchemaChange} outputSchema={outputSchema}
      />
    </div>
  );
}
