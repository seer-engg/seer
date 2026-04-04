import { IfElseBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';

type IfElseBlockSectionProps = BlockSectionProps<IfElseBlockConfig>;

export function IfElseBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
}: IfElseBlockSectionProps) {

  return (
    <div className="space-y-4">
      <DynamicFormField
        name="condition"
        label="Condition Expression"
        description="Boolean expression using ${variable} syntax. Use ${variable} expressions referencing upstream outputs or workflow inputs."
        value={config.condition || ''}
        onChange={value => setConfig(prev => ({ ...prev, condition: value as string }))}
        placeholder="e.g., ${alias.output} > 0"
        def={{ type: 'string', multiline: false }}
        templateAutocomplete={templateAutocomplete}
        className="font-mono"
        error={validationErrors['condition']}
      />
    </div>
  );
}

