import { useState, useEffect, useMemo } from 'react';
import { ArrowLeft, Loader2, Zap } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { Badge } from '@/components/ui/badge';
import { TemplateConfigForm } from './TemplateConfigForm';
import { IntegrationRequirements } from './IntegrationRequirements';
import type { TemplateDetail, TemplateRequirementsResponse } from '@/types/templates';

export interface TemplateDetailViewProps {
  template: TemplateDetail;
  requirements: TemplateRequirementsResponse | null;
  isLoadingRequirements: boolean;
  isInstantiating: boolean;
  onBack: () => void;
  onInstantiate: (name: string, config: Record<string, unknown>) => void;
}

function getDefaultConfig(fields: TemplateDetail['config_fields']): Record<string, unknown> {
  const defaults: Record<string, unknown> = {};
  fields.forEach((field) => {
    if (field.default !== undefined) {
      defaults[field.name] = field.default;
    }
  });
  return defaults;
}

interface TemplateSetupSectionProps {
  template: TemplateDetail;
  requirements: TemplateRequirementsResponse | null;
  isLoadingRequirements: boolean;
  workflowName: string;
  onWorkflowNameChange: (name: string) => void;
  configValues: Record<string, unknown>;
  onConfigChange: (name: string, value: unknown) => void;
  canInstantiate: boolean;
  isInstantiating: boolean;
  onInstantiate: () => void;
}

function TemplateSetupSection({
  template,
  requirements,
  isLoadingRequirements,
  workflowName,
  onWorkflowNameChange,
  configValues,
  onConfigChange,
  canInstantiate,
  isInstantiating,
  onInstantiate,
}: TemplateSetupSectionProps) {
  return (
    <div className="rounded-xl border bg-card p-6 space-y-5">
      <div>
        <h2 className="text-lg font-semibold mb-1">Set up this template</h2>
        <p className="text-sm text-muted-foreground">Configure and create a workflow from this template.</p>
      </div>

      {/* Required Integrations */}
      <div>
        <h3 className="text-sm font-semibold mb-2">Required Integrations</h3>
        {isLoadingRequirements ? (
          <div className="space-y-2">
            <Skeleton className="h-12 w-full" />
          </div>
        ) : requirements ? (
          <IntegrationRequirements requirements={requirements} />
        ) : (
          <p className="text-sm text-muted-foreground">
            This template has no integration requirements.
          </p>
        )}
      </div>

      {/* Config fields */}
      {template.config_fields.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold mb-2">Configuration</h3>
          <TemplateConfigForm
            fields={template.config_fields}
            values={configValues}
            onChange={onConfigChange}
          />
        </div>
      )}

      {/* Workflow name + CTA */}
      <div className="pt-3 border-t space-y-4">
        <div className="space-y-1.5">
          <Label htmlFor="workflow-name">
            Workflow Name <span className="text-destructive">*</span>
          </Label>
          <Input
            id="workflow-name"
            value={workflowName}
            onChange={(e) => onWorkflowNameChange(e.target.value)}
            placeholder="Enter a name for your workflow"
          />
        </div>

        <Button
          onClick={onInstantiate}
          disabled={!canInstantiate || isInstantiating}
          size="lg"
          className="w-full"
        >
          {isInstantiating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Creating Workflow...
            </>
          ) : (
            <>
              <Zap className="w-4 h-4 mr-2" />
              Create Workflow
            </>
          )}
        </Button>
      </div>
    </div>
  );
}

function TemplateHero({ template }: { template: TemplateDetail }) {
  return (
    <div className="rounded-xl border bg-card p-6 mb-6">
      <div className="flex items-start gap-4">
        {template.icon ? (
          <span className="text-4xl mt-0.5">{template.icon}</span>
        ) : (
          <div className="w-14 h-14 rounded-xl bg-primary/10 flex items-center justify-center shrink-0">
            <span className="text-primary text-2xl font-bold">
              {template.name.charAt(0).toUpperCase()}
            </span>
          </div>
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 mb-1">
            <h1 className="text-2xl font-bold truncate">{template.name}</h1>
            <Badge variant="secondary" className="shrink-0 capitalize">{template.category}</Badge>
          </div>
          <p className="text-muted-foreground leading-relaxed mt-1">
            {template.description}
          </p>
        </div>
      </div>

      {template.preview_image_url && (
        <div className="rounded-lg overflow-hidden border mt-5">
          <img
            src={template.preview_image_url}
            alt={`Preview of ${template.name}`}
            className="w-full h-auto"
          />
        </div>
      )}
    </div>
  );
}

export function TemplateDetailView({
  template,
  requirements,
  isLoadingRequirements,
  isInstantiating,
  onBack,
  onInstantiate,
}: TemplateDetailViewProps) {
  const [workflowName, setWorkflowName] = useState(template.name);
  const [configValues, setConfigValues] = useState<Record<string, unknown>>(() =>
    getDefaultConfig(template.config_fields),
  );

  useEffect(() => {
    setWorkflowName(template.name);
    setConfigValues(getDefaultConfig(template.config_fields));
  }, [template]);

  const isConfigValid = useMemo(() => {
    return template.config_fields.every((field) => {
      if (!field.required) return true;
      const value = configValues[field.name];
      return value !== undefined && value !== null && value !== '';
    });
  }, [template.config_fields, configValues]);

  const canInstantiate = workflowName.trim() !== '' && isConfigValid;

  const handleConfigChange = (name: string, value: unknown) => {
    setConfigValues((prev) => ({ ...prev, [name]: value }));
  };

  const handleInstantiate = () => {
    if (canInstantiate) {
      onInstantiate(workflowName.trim(), configValues);
    }
  };

  return (
    <div className="max-w-3xl mx-auto">
      {/* Back button */}
      <button
        onClick={onBack}
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-6"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Templates
      </button>

      <TemplateHero template={template} />

      <TemplateSetupSection
        template={template}
        requirements={requirements}
        isLoadingRequirements={isLoadingRequirements}
        workflowName={workflowName}
        onWorkflowNameChange={setWorkflowName}
        configValues={configValues}
        onConfigChange={handleConfigChange}
        canInstantiate={canInstantiate}
        isInstantiating={isInstantiating}
        onInstantiate={handleInstantiate}
      />
    </div>
  );
}
