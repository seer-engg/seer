import type { TemplateAutocompleteControls } from '../types';
import { DynamicFormField } from './DynamicFormField';
import type { DynamicFieldDef } from './DynamicFormField';

export interface ToolParamDefinition extends DynamicFieldDef {
  required?: boolean;
}

export interface ParamInputFactoryProps {
  paramName: string;
  paramDef: ToolParamDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  templateAutocomplete: TemplateAutocompleteControls;
}

/**
 * Factory component that renders the appropriate param input based on param type
 */
export function ParamInputFactory({
  paramName,
  paramDef,
  value,
  onChange,
  templateAutocomplete,
}: ParamInputFactoryProps) {
  return (
    <DynamicFormField
      name={paramName}
      label={paramName}
      description={paramDef.description as string | undefined}
      value={value}
      onChange={onChange}
      def={paramDef}
      templateAutocomplete={templateAutocomplete}
    />
  );
}
