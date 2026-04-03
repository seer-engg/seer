/**
 * Utility functions for Logo.dev integration
 * Extracts company/service names from tool names and generates logo URLs
 */

/**
 * Extract service name from tool name
 * Examples:
 *   google_sheets_write → google
 *   supabase_table_query → supabase
 *   github_pr_create → github
 *   gmail_read_emails → gmail
 */
export function extractServiceName(toolName: string): string | null {
  if (!toolName) return null;

  const normalized = toolName.toLowerCase().trim();

  // Common service name patterns
  const patterns = [
    /^(google)_/,       // google_sheets_write, google_drive_upload
    /^(github)_/,       // github_pr_create, github_issue_list
    /^(gmail)_/,        // gmail_read_emails, gmail_send_email
    /^(supabase)_/,     // supabase_table_query, supabase_table_insert
    /^(slack)_/,        // slack_send_message
    /^(notion)_/,       // notion_create_page
    /^(airtable)_/,     // airtable_list_records
    /^(stripe)_/,       // stripe_create_payment
    /^(hubspot)_/,      // hubspot_create_contact
    /^(salesforce)_/,   // salesforce_query
    /^(linear)_/,       // linear_create_issue
    /^(asana)_/,        // asana_create_task
    /^(trello)_/,       // trello_create_card
    /^(jira)_/,         // jira_create_issue
    /^(zendesk)_/,      // zendesk_create_ticket
  ];

  for (const pattern of patterns) {
    const match = normalized.match(pattern);
    if (match) {
      return match[1];
    }
  }

  // If no pattern matches, try to extract the first word before underscore
  const firstPart = normalized.split('_')[0];
  if (firstPart && firstPart.length > 2) {
    return firstPart;
  }

  return null;
}

/**
 * Generate Logo.dev URL for a service
 * @param serviceName - The service name (e.g., "google", "github")
 * @param size - Logo size in pixels (default: 32)
 * @returns Logo.dev URL or null if token is not configured
 */
export function getLogoUrl(serviceName: string | null, size: number = 32): string | null {
  if (!serviceName) return null;

  const token = import.meta.env.VITE_LOGO_DEV_TOKEN;
  if (!token) {
    console.warn('VITE_LOGO_DEV_TOKEN is not configured');
    return null;
  }

  return `https://img.logo.dev/${serviceName}.com?token=${token}&size=${size}`;
}

/**
 * Get logo URL directly from tool name
 * @param toolName - The tool name (e.g., "google_sheets_write")
 * @param size - Logo size in pixels (default: 32)
 * @returns Logo.dev URL or null if service cannot be extracted or token is not configured
 */
export function getToolLogoUrl(toolName: string, size: number = 32): string | null {
  const serviceName = extractServiceName(toolName);
  return getLogoUrl(serviceName, size);
}

/**
 * Map integration_type to Logo.dev domain
 * Some integrations have different domains than their type name
 *
 * Note: Google products (gmail, google_drive, google_sheets, etc.) are handled
 * separately with custom SVG icons in IntegrationIcon component because
 * Logo.dev can't distinguish Google sub-products (they all return the "G" logo)
 */
const INTEGRATION_TYPE_TO_DOMAIN: Record<string, string> = {
  github: 'github.com',
  discord: 'discord.com',
  linkedin: 'linkedin.com',
  supabase: 'supabase.com',
  slack: 'slack.com',
  notion: 'notion.so',
  airtable: 'airtable.com',
  stripe: 'stripe.com',
  hubspot: 'hubspot.com',
  salesforce: 'salesforce.com',
  linear: 'linear.app',
  asana: 'asana.com',
  trello: 'trello.com',
  jira: 'atlassian.com',
  zendesk: 'zendesk.com',
  twitter: 'x.com',
  dropbox: 'dropbox.com',
  figma: 'figma.com',
  intercom: 'intercom.com',
  mailchimp: 'mailchimp.com',
  twilio: 'twilio.com',
  sendgrid: 'sendgrid.com',
  shopify: 'shopify.com',
  zoom: 'zoom.us',
  microsoft: 'microsoft.com',
  outlook: 'outlook.com',
  teams: 'teams.microsoft.com',
};

/**
 * Get Logo.dev URL for an integration type
 * @param integrationType - The integration type (e.g., "gmail", "google_drive")
 * @param size - Logo size in pixels (default: 32)
 * @returns Logo.dev URL or null if token is not configured
 */
export function getIntegrationLogoUrl(integrationType: string, size: number = 32): string | null {
  if (!integrationType) return null;

  const token = import.meta.env.VITE_LOGO_DEV_TOKEN;
  if (!token) return null;

  const normalized = integrationType.toLowerCase().trim();
  const domain = INTEGRATION_TYPE_TO_DOMAIN[normalized] ?? `${normalized}.com`;

  return `https://img.logo.dev/${domain}?token=${token}&size=${size}`;
}
