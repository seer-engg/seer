/**
 * Resource Picker Question Component
 *
 * Wrapper component that adapts the ClarificationQuestion format to the
 * ResourcePicker component used throughout the workflow builder.
 *
 * Handles the mapping between:
 * - Backend question format (question_id, depends_on as question_id)
 * - ResourcePicker format (config object, dependsOnValues as field name map)
 */
import { ResourcePicker } from '@/components/workflows/dialogs/ResourcePicker';
import type { ResourcePickerConfig } from '@/components/workflows/dialogs/ResourcePicker/types';
import type { ClarificationQuestion } from '@/types/discovery';

/**
 * Hardcoded endpoint overrides for resource types that don't follow
 * the default /api/integrations/resources/{provider}/{resource_type} pattern.
 *
 * TODO: This is a temporary fix. The proper solution is for the backend
 * to include the `endpoint` field in ClarificationQuestion responses.
 */
const ENDPOINT_OVERRIDES: Record<string, string> = {
  knowledge_base: '/api/v1/knowledge-bases',
};

interface ResourcePickerQuestionProps {
  /** The clarification question from the backend */
  question: ClarificationQuestion;
  /** Currently selected resource value */
  selectedValue: string;
  /** Callback when a resource is selected */
  onValueChange: (value: string, displayName?: string) => void;
  /** Whether the form is in a loading/submitting state */
  isLoading: boolean;
  /** Resolved dependency values (field name -> selected value) */
  dependsOnValues?: Record<string, string>;
}

/**
 * Builds a ResourcePickerConfig from a ClarificationQuestion
 */
function buildConfig(question: ClarificationQuestion): ResourcePickerConfig {
  return {
    resource_type: question.resource_type || '',
    display_field: question.display_field,
    value_field: question.value_field,
    search_enabled: question.search_enabled,
    hierarchy: question.hierarchy,
    // Map depends_on_field to config.depends_on (the API parameter name)
    depends_on: question.depends_on_field,
    // Use hardcoded endpoint override if available for this resource type
    endpoint: ENDPOINT_OVERRIDES[question.resource_type || ''],
  };
}

export function ResourcePickerQuestion({
  question,
  selectedValue,
  onValueChange,
  isLoading,
  dependsOnValues,
}: ResourcePickerQuestionProps) {
  const config = buildConfig(question);

  return (
    <ResourcePicker
      config={config}
      provider={question.provider}
      value={selectedValue || undefined}
      onChange={onValueChange}
      placeholder={`Select ${question.resource_type?.replace(/_/g, ' ') || 'resource'}...`}
      disabled={isLoading}
      dependsOnValues={dependsOnValues}
      className="text-sm"
    />
  );
}
