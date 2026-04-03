import { ForLoopBlockConfig, BlockSectionProps } from '../types';
import { DynamicFormField } from '../widgets/DynamicFormField';

type ForLoopBlockSectionProps = BlockSectionProps<ForLoopBlockConfig>;

export function ForLoopBlockSection({
  config,
  setConfig,
  templateAutocomplete,
  validationErrors = {},
}: ForLoopBlockSectionProps) {
  const arrayVariable = config.array_variable || config.array_var || 'items';
  const legacyLiteralItems = Array.isArray(config.array_literal) ? config.array_literal : [];

  return (
    <div className="space-y-4">
      <DynamicFormField
        name="array_variable"
        label="Array source variable"
        description="Reference an array from upstream blocks using ${variable} syntax"
        value={arrayVariable}
        onChange={value => {
          setConfig(prev => {
            const next: ForLoopBlockConfig = { ...prev, array_variable: String(value), array_var: undefined };
            delete next.array_literal;
            delete next.array_mode;
            return next;
          });
        }}
        placeholder="e.g., ${blockAlias.output}"
        def={{ type: 'string' }}
        templateAutocomplete={templateAutocomplete}
        className="font-mono"
        error={validationErrors['array_variable']}
      />

      {legacyLiteralItems.length > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
          Legacy manual lists ({legacyLiteralItems.length} item
          {legacyLiteralItems.length === 1 ? '' : 's'}) are read-only. Point this loop to a variable
          above to migrate.
        </div>
      )}

      <DynamicFormField
        name="item_var"
        label="Item variable name"
        description="Variable name for each item in the loop. Downstream blocks can reference this variable."
        defaultValue="item"
        value={config.item_var || 'item'}
        onChange={value => setConfig(prev => ({ ...prev, item_var: String(value) }))}
        placeholder="e.g., email"
        def={{ type: 'string' }}
        templateAutocomplete={templateAutocomplete}
        error={validationErrors['item_var']}
      />

      <div className="rounded-md border border-dashed border-muted-foreground/40 p-3 text-xs text-muted-foreground">
        <p className="font-medium text-foreground">Loop routing</p>
        <p className="mt-1">
          Use the <span className="font-medium text-foreground">Loop</span> handle to connect blocks
          that should run for every item. Once the array is exhausted, the workflow continues through
          the <span className="font-medium text-foreground">Exit</span> handle exactly once.
        </p>
      </div>
    </div>
  );
}

