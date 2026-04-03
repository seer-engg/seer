/**
 * Types for the Workflow Templates feature.
 * These types mirror the backend API responses for template endpoints.
 */

/**
 * Represents a required integration for a template.
 * Used to indicate what OAuth connections are needed.
 */
export interface RequiredIntegration {
  provider: string;
  integration_type: string;
  reason: string;
}

/**
 * Configuration field definition for template customization.
 * Templates can define dynamic fields that users fill out before instantiation.
 */
export interface TemplateConfigField {
  name: string;
  label: string;
  type: 'string' | 'number' | 'boolean' | 'select';
  required: boolean;
  default?: unknown;
  description?: string;
  options?: { value: string; label: string }[];
}

/**
 * Summary information for a template, used in gallery listings.
 */
export interface TemplateSummary {
  template_id: string;
  slug: string;
  name: string;
  description: string;
  category: string;
  tags: string[];
  icon?: string;
  is_featured: boolean;
  usage_count: number;
  required_integrations: RequiredIntegration[];
}

/**
 * Full template details including spec and config fields.
 * Returned when fetching a single template by slug.
 */
export interface TemplateDetail extends TemplateSummary {
  spec: Record<string, unknown>;
  config_fields: TemplateConfigField[];
  preview_image_url?: string;
}

/**
 * Template category with metadata.
 */
export interface TemplateCategory {
  slug: string;
  name: string;
  description?: string;
  template_count: number;
}

/**
 * Response from the template requirements endpoint.
 * Shows which integrations the user has connected vs missing.
 */
export interface TemplateRequirementsResponse {
  connected: RequiredIntegration[];
  missing: RequiredIntegration[];
}

/**
 * Request body for instantiating a template into a new workflow.
 */
export interface TemplateInstantiateRequest {
  name: string;
  config: Record<string, unknown>;
  provider_connections: Record<string, number>;
}

/**
 * Response after successfully instantiating a template.
 */
export interface TemplateInstantiateResponse {
  workflow_id: string;
  name: string;
  missing_integrations: RequiredIntegration[];
}

/**
 * API response wrapper for template list.
 */
export interface TemplateListResponse {
  items: TemplateSummary[];
  total: number;
}

/**
 * API response wrapper for categories list.
 */
export interface TemplateCategoriesResponse {
  items: TemplateCategory[];
}

// =============================================================================
// Admin Types
// =============================================================================

/**
 * Request to create a new template from an existing workflow.
 */
export interface TemplateCreateRequest {
  workflow_id: string;
  slug: string;
  name: string;
  description: string;
  category: string;
  tags?: string[];
  required_integrations?: RequiredIntegration[];
  icon?: string;
  preview_image_url?: string;
  is_featured?: boolean;
}

/**
 * Request to update an existing template.
 */
export interface TemplateUpdateRequest {
  slug?: string;
  name?: string;
  description?: string;
  category?: string;
  tags?: string[];
  required_integrations?: RequiredIntegration[];
  icon?: string;
  preview_image_url?: string;
  is_featured?: boolean;
}

/**
 * Admin template response (includes timestamps).
 */
export interface TemplateAdminResponse extends TemplateDetail {
  source: string;
  created_at: string;
  updated_at: string;
}

/**
 * API response wrapper for admin template list.
 */
export interface TemplateAdminListResponse {
  items: TemplateAdminResponse[];
  total: number;
}

/**
 * Template categories enum values for admin forms.
 */
export const TEMPLATE_CATEGORIES = [
  { value: 'marketing', label: 'Marketing' },
  { value: 'customer_support', label: 'Customer Support' },
  { value: 'sales', label: 'Sales' },
  { value: 'productivity', label: 'Productivity' },
  { value: 'engineering', label: 'Engineering' },
  { value: 'operations', label: 'Operations' },
  { value: 'hr', label: 'HR' },
  { value: 'other', label: 'Other' },
] as const;
