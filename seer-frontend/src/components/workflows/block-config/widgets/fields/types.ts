import type { TemplateAutocompleteControls, ResourcePickerConfig } from '../../types';

/**
 * Base props shared by all field renderers
 */
export interface BaseFieldProps {
  /** Field ID for htmlFor */
  id: string;
  /** Current field value */
  value: unknown;
  /** Change handler */
  onChange: (value: unknown) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Whether to show error styling */
  showError: boolean;
  /** Template autocomplete controls */
  templateAutocomplete: TemplateAutocompleteControls;
}

/**
 * Props for ResourcePickerField component
 */
export interface ResourcePickerFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange'> {
  value: unknown;
  onChange: (value: unknown) => void;
  config: ResourcePickerConfig;
  provider?: string;
  dependsOnValues?: Record<string, string>;
  fieldLabel: string;
  fieldName: string;
  onResourceLabelChange?: (fieldName: string, label?: string) => void;
  type: string;
}

/**
 * Props for EnumSelectField component
 */
export interface EnumSelectFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange'> {
  value: unknown;
  onChange: (value: string) => void;
  enumOptions: string[];
  enumLabels?: string[];
}

/**
 * Props for BooleanField component
 */
export interface BooleanFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange'> {
  value: unknown;
  onChange: (value: boolean) => void;
}

/**
 * Props for NumericField component
 */
export interface NumericFieldProps extends BaseFieldProps {
  type: 'integer' | 'number';
  minimum?: number;
  maximum?: number;
}

/**
 * Props for JsonField component
 */
export interface JsonFieldProps extends BaseFieldProps {
  type: 'array' | 'object';
  rows?: number;
}

/**
 * Props for TextField component
 */
export interface TextFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange'> {
  value: unknown;
  onChange: (value: string) => void;
  multiline: boolean;
  rows?: number;
}

/**
 * Editor sub-mode within a content mode
 * - 'visual': Tiptap WYSIWYG editor (simple mode only)
 * - 'html-source': Raw HTML textarea
 * - 'html-preview': Sandboxed iframe preview (full-html mode only)
 */
export type EditorMode = 'visual' | 'html-source' | 'html-preview';

/**
 * User-selected content mode
 * - 'simple': Uses Tiptap for editing, HTML is parsed through Tiptap's schema
 * - 'full-html': Raw HTML mode, preserves all HTML without schema parsing
 */
export type ContentMode = 'simple' | 'full-html';

/**
 * Props for RichTextField component
 */
export interface RichTextFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange' | 'showError'> {
  value: string;
  onChange: (value: string) => void;
  /** Output format: 'html' or 'markdown' from tool schema's x-rich-text-output */
  outputFormat?: 'html' | 'markdown';
  /** Allowed formatting features from tool schema's x-rich-text-features */
  features?: string[];
  /** Character limit from tool schema's maxLength */
  charLimit?: number;
  /** Whether to show error styling */
  showError?: boolean;
  /** Number of rows for editor height */
  rows?: number;
}

/**
 * Props for FileInputField component
 * Note: File inputs have their own autocomplete using availableFileVariables, so templateAutocomplete is excluded
 */
export interface FileInputFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange' | 'templateAutocomplete'> {
  value: import('../../helpers/fileInputSchema').FileInputValue;
  onChange: (value: import('../../helpers/fileInputSchema').FileInputValue) => void;
  /** Available file variables from upstream nodes */
  availableFileVariables?: import('../../helpers/discoverFileVariables').FileVariable[];
  /** Optional MIME type filter for file picker */
  mimeTypeFilter?: string;
}

/**
 * Props for FileInputArrayField component
 * Note: File inputs have their own autocomplete using availableFileVariables, so templateAutocomplete is excluded
 */
export interface FileInputArrayFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange' | 'templateAutocomplete'> {
  value: import('../../helpers/fileInputSchema').FileInputValue[];
  onChange: (value: import('../../helpers/fileInputSchema').FileInputValue[]) => void;
  /** Available file variables from upstream nodes */
  availableFileVariables?: import('../../helpers/discoverFileVariables').FileVariable[];
  /** Optional MIME type filter for file picker */
  mimeTypeFilter?: string;
}
