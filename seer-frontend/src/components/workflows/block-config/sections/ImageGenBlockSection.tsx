import { ImageGenBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';

const IMAGE_GEN_MODELS = [
  { value: 'sourceful/riverflow-v2-fast', label: 'Riverflow V2 Fast' },
  { value: 'google/gemini-2.5-flash-image', label: 'Gemini 2.5 Flash Image' },
];

const IMAGE_SIZE_OPTIONS = [
  '1024x1024',
  '512x512',
  '1024x768',
  '768x1024',
];

export function ImageGenBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
}: BlockSectionProps<ImageGenBlockConfig>) {
  return (
    <div className="space-y-2">
      <DynamicFormField
        name="prompt"
        label="Prompt"
        description="Describe the image you want to generate"
        required
        value={config.prompt || ''}
        onChange={value => setConfig(prev => ({ ...prev, prompt: value as string }))}
        def={{ type: 'string', multiline: true }}
        templateAutocomplete={templateAutocomplete}
        rows={6}
        error={validationErrors['prompt']}
      />

      <DynamicFormField
        name="model"
        label="Model"
        description="Choose the image generation model"
        value={config.model || IMAGE_GEN_MODELS[0].value}
        onChange={value => setConfig(prev => ({ ...prev, model: String(value) }))}
        def={{ type: 'string', enum: IMAGE_GEN_MODELS.map(m => m.value) }}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['model']}
      />

      <DynamicFormField
        name="size"
        label="Size"
        description="Image dimensions"
        value={config.size || '1024x1024'}
        onChange={value => setConfig(prev => ({ ...prev, size: String(value) }))}
        def={{ type: 'string', enum: IMAGE_SIZE_OPTIONS }}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['size']}
      />
    </div>
  );
}
