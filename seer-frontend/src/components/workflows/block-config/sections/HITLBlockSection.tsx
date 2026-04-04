import { HitlBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';
import { HITLDisplayEditor } from './HITLDisplayEditor';
import { HITLInputsEditor } from './HITLInputsEditor';
import { HITLDeliveryChannelsEditor } from './HITLDeliveryChannelsEditor';
import { generateHitlOutputSchema } from '../helpers/hitlOutputSchema';

type HITLBlockSectionProps = BlockSectionProps<HitlBlockConfig>;

/**
 * Configuration section for HITL (Human-in-the-Loop) blocks.
 *
 * Allows users to configure:
 * - Title and description for the approval prompt
 * - Display items to show workflow data to the reviewer
 * - Input fields to collect human responses
 * - Timeout for automatic workflow cancellation
 * - Delivery channels (platform and/or email notifications)
 */
export function HITLBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
}: HITLBlockSectionProps) {
  return (
    <div className="space-y-4">
      <DynamicFormField
        name="title"
        label="Title"
        description="Heading shown to the human reviewer"
        required
        value={config.title || ''}
        onChange={(value) => setConfig((prev) => ({ ...prev, title: value as string }))}
        placeholder="e.g., Approval Required"
        def={{ type: 'string', multiline: false }}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['title']}
      />

      <DynamicFormField
        name="description"
        label="Description"
        description="Additional context or instructions for the reviewer"
        value={config.description || ''}
        onChange={(value) => setConfig((prev) => ({ ...prev, description: value as string }))}
        placeholder="e.g., Please review the following request and approve or reject."
        def={{ type: 'string', multiline: true }}
        templateAutocomplete={templateAutocomplete}
        rows={3}
        error={validationErrors['description']}
      />

      <div className="border-t pt-4">
        <HITLDisplayEditor
          display={config.display || []}
          onChange={(display) => setConfig((prev) => ({ ...prev, display }))}
          templateAutocomplete={templateAutocomplete}
        />
      </div>

      <div className="border-t pt-4">
        <HITLInputsEditor
          inputs={config.inputs || []}
          onChange={(inputs) => {
            const output_schema = generateHitlOutputSchema(inputs);
            setConfig((prev) => ({ ...prev, inputs, output_schema }));
          }}
          templateAutocomplete={templateAutocomplete}
        />
      </div>

      <div className="border-t pt-4">
        <DynamicFormField
          name="timeout_seconds"
          label="Timeout (seconds)"
          description="How long to wait for a response before the workflow times out. Default is 24 hours (86400 seconds)."
          value={config.timeout_seconds ?? 86400}
          onChange={(value) =>
            setConfig((prev) => ({
              ...prev,
              timeout_seconds: typeof value === 'string' ? parseInt(value, 10) : (value as number),
            }))
          }
          placeholder="86400"
          def={{ type: 'number', minimum: 60, maximum: 604800 }}
          templateAutocomplete={templateAutocomplete}
          error={validationErrors['timeout_seconds']}
        />
      </div>

      <div className="border-t pt-4">
        <HITLDeliveryChannelsEditor
          channels={config.delivery_channels ?? [{ type: 'platform' }]}
          onChange={(delivery_channels) => setConfig((prev) => ({ ...prev, delivery_channels }))}
          templateAutocomplete={templateAutocomplete}
        />
      </div>
    </div>
  );
}
