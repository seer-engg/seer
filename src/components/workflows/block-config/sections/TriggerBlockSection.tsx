import { Node } from '@xyflow/react';
import { Button } from '@/components/ui/button';
import { Trash2 } from 'lucide-react';

import type { TemplateAutocompleteControls } from '../types';
import type { WorkflowNodeData } from '../../types';

// Import hooks
import { useTriggerConfig } from '../../blocks/TriggerBlockNode/triggers/useTriggerConfig';
import { useBaseTrigger } from '../../blocks/TriggerBlockNode/components/useBaseTrigger';

// Import details sections
import { WebhookDetailsSection } from '../../blocks/TriggerBlockNode/triggers/WebhookTriggerConfig';
import { FormDetailsSection } from '../../blocks/TriggerBlockNode/triggers/FormDetailsSection';
import { GmailDetailsSection } from '../../blocks/TriggerBlockNode/triggers/GmailTriggerConfig';
import { DiscordDetailsSection } from '../../blocks/TriggerBlockNode/triggers/DiscordTriggerConfig';
import { SlackDetailsSection } from '../../blocks/TriggerBlockNode/triggers/SlackTriggerConfig';
import { GoogleCalendarDetailsSection } from '../../blocks/TriggerBlockNode/triggers/GoogleCalendarTriggerConfig';
import { GoogleSheetsDetailsSection } from '../../blocks/TriggerBlockNode/triggers/GoogleSheetsTriggerConfig';
import { MeetingsEADetailsSection } from '../../blocks/TriggerBlockNode/triggers/MeetingsEATriggerConfig';

// Import config sections
import {
  ExpandedConfigSection,
  CronDetailsSectionFromConfig,
} from '../../blocks/TriggerBlockNode/components/TriggerBlockNodeView';

interface TriggerBlockSectionProps {
  node: Node<WorkflowNodeData>;
  templateAutocomplete: TemplateAutocompleteControls;
  validationErrors?: Record<string, string>;
  readOnly?: boolean;
}

export function TriggerBlockSection({
  node,
  templateAutocomplete,
  validationErrors = {},
  readOnly,
}: TriggerBlockSectionProps) {
  // Extract triggerMeta from node
  const triggerMeta = node.data.triggerMeta;

  if (!triggerMeta) {
    return (
      <div className="p-4 text-center text-muted-foreground">
        Trigger metadata unavailable
      </div>
    );
  }

  return (
    <TriggerBlockSectionWithMeta
      node={node}
      triggerMeta={triggerMeta}
      templateAutocomplete={templateAutocomplete}
      validationErrors={validationErrors}
      readOnly={readOnly}
    />
  );
}

interface TriggerBlockSectionWithMetaProps extends TriggerBlockSectionProps {
  triggerMeta: NonNullable<WorkflowNodeData['triggerMeta']>;
  readOnly?: boolean;
}

function renderTriggerDetailsSection(
  triggerId: string,
  kind: ReturnType<typeof useTriggerConfig>['kind'],
  triggerKey: string,
  subscription: WorkflowNodeData['triggerMeta']['subscription'],
  state: ReturnType<typeof useTriggerConfig>['state'],
) {
  if (kind === 'webhook') {
    return (
      <WebhookDetailsSection
        subscription={subscription}
        setDataSchema={state.kind === 'webhook' ? state.setDataSchema : undefined}
      />
    );
  }

  if (kind === 'form') {
    return (
      <FormDetailsSection
        subscription={subscription}
        setDataSchema={state.kind === 'webhook' ? state.setDataSchema : undefined}
      />
    );
  }

  if (kind === 'gmail') {
    return <GmailDetailsSection triggerKey={triggerKey} triggerId={triggerId} />;
  }

  if (kind === 'discord') {
    return <DiscordDetailsSection triggerKey={triggerKey} triggerId={triggerId} />;
  }

  if (kind === 'slack') {
    return <SlackDetailsSection triggerKey={triggerKey} triggerId={triggerId} />;
  }

  if (kind === 'calendar') {
    return <GoogleCalendarDetailsSection triggerKey={triggerKey} triggerId={triggerId} />;
  }

  if (kind === 'google_sheets') {
    return <GoogleSheetsDetailsSection triggerKey={triggerKey} triggerId={triggerId} />;
  }

  if (kind === 'meetings_ea') {
    return <MeetingsEADetailsSection />;
  }

  if (kind === 'cron' && state.kind === 'dynamic') {
    return <CronDetailsSectionFromConfig configValues={state.configValues} />;
  }

  return null;
}

function TriggerBlockSectionWithMeta({
  node,
  triggerMeta,
  templateAutocomplete,
  validationErrors: _validationErrors = {},
  readOnly,
}: TriggerBlockSectionWithMetaProps) {
  const triggerKey = (node.data.config?.triggerKey as string | undefined) ?? triggerMeta.subscription?.key ?? '';

  // Use existing hooks (same as TriggerBlockNodeView)
  const { kind, state, quickOptions, descriptor, subscription } = useTriggerConfig(node.id, triggerKey, triggerMeta, { readOnly });
  const {
    bindingState,
    handleBindingModeChange,
    handleBindingValueChange,
    bindingQuickInsert,
    handleDelete,
    isDraft,
  } = useBaseTrigger({
    subscription,
    handlers: triggerMeta.handlers,
  });

  return (
    <div className="space-y-4">
      {renderTriggerDetailsSection(node.id, kind, triggerKey, subscription, state)}

      {/* Configuration forms - reuse ExpandedConfigSection */}
      <ExpandedConfigSection
        triggerKind={kind}
        config={state}
        descriptor={descriptor}
        subscription={subscription}
        templateAutocomplete={templateAutocomplete}
        bindingState={bindingState}
        handleBindingModeChange={handleBindingModeChange}
        handleBindingValueChange={handleBindingValueChange}
        bindingQuickInsert={bindingQuickInsert}
        quickOptions={quickOptions}
      />

      {/* Delete button */}
      <div className="pt-4 border-t">
        <Button
          variant="destructive"
          size="sm"
          onClick={handleDelete}
          disabled={isDraft}
          className="w-full"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete Trigger
        </Button>
      </div>
    </div>
  );
}
