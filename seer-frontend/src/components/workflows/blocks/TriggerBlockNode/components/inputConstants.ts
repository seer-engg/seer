import type { InputDef } from '@/types/workflow-spec';

export const INPUT_NAME_REGEX = /^[A-Za-z_][A-Za-z0-9_]*$/;

export const INPUT_TYPE_OPTIONS: Array<{ label: string; value: InputDef['type'] }> = [
  { label: 'Text', value: 'string' },
  { label: 'Number', value: 'number' },
  { label: 'Integer', value: 'integer' },
  { label: 'Boolean', value: 'boolean' },
  { label: 'Object', value: 'object' },
  { label: 'Array', value: 'array' },
];
