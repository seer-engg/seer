import { EABotBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';

type EABotBlockSectionProps = BlockSectionProps<EABotBlockConfig>;

/**
 * Configuration section for meetingsEA blocks.
 *
 * Allows users to configure:
 * - Meeting URL (supports template expressions)
 * - Bot display name
 * - Recording mode (speaker view, gallery view, audio only)
 * - Timeout for max wait time
 */
export function EABotBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
}: EABotBlockSectionProps) {
  return (
    <div className="space-y-4">
      <DynamicFormField
        name="meeting_url"
        label="Meeting URL"
        description="Google Meet, Zoom, or Microsoft Teams link"
        required
        value={config.meeting_url || ''}
        onChange={(value) => setConfig((prev) => ({ ...prev, meeting_url: value as string }))}
        placeholder="e.g., ${trigger.data.conference_link}"
        def={{ type: 'string', multiline: false }}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['meeting_url']}
      />

      <DynamicFormField
        name="bot_name"
        label="Bot Name"
        description="Display name shown in the meeting"
        value={config.bot_name || 'meetingsEA'}
        onChange={(value) => setConfig((prev) => ({ ...prev, bot_name: value as string }))}
        placeholder="meetingsEA"
        def={{ type: 'string', multiline: false }}
        templateAutocomplete={templateAutocomplete}
      />

      <DynamicFormField
        name="recording_mode"
        label="Recording Mode"
        description="How to record the meeting"
        value={config.recording_mode || 'speaker_view'}
        onChange={(value) => setConfig((prev) => ({ ...prev, recording_mode: value as EABotBlockConfig['recording_mode'] }))}
        def={{
          type: 'string',
          enum: ['speaker_view', 'gallery_view', 'audio_only'],
        }}
        templateAutocomplete={templateAutocomplete}
      />

      <DynamicFormField
        name="timeout_seconds"
        label="Timeout (seconds)"
        description="Max time to wait for the meeting to end. Default: 14400 (4 hours)."
        value={config.timeout_seconds ?? 14400}
        onChange={(value) => setConfig((prev) => ({ ...prev, timeout_seconds: Number(value) || 14400 }))}
        def={{ type: 'integer', minimum: 60, maximum: 86400 }}
        templateAutocomplete={templateAutocomplete}
      />

      <div className="rounded-md border border-dashed p-3 bg-muted/40">
        <p className="text-xs text-muted-foreground">
          The bot will join the meeting, record and transcribe it. The workflow pauses until the meeting ends,
          then continues with the transcript and recording URL available as outputs.
        </p>
      </div>
    </div>
  );
}
