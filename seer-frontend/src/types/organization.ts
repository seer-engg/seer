/**
 * Organization Types
 *
 * Types for the multi-tenant team/workspace system.
 * Every user has a personal organization, and can be members of team organizations.
 */

// =============================================================================
// Core Types
// =============================================================================

/**
 * Organization roles determine what actions a user can perform within an org.
 *
 * - owner: Full control, including billing and deletion
 * - admin: Can manage members and all workflows
 * - user: Can create and manage own workflows
 * - consultant: External access, workflows need approval
 */
export type OrganizationRole = 'owner' | 'admin' | 'user' | 'consultant';

/**
 * Organization type distinguishes between personal workspaces and teams.
 *
 * - personal: Auto-created for each user, cannot be deleted
 * - team: Created by users with active subscriptions, supports multiple members
 */
export type OrganizationType = 'personal' | 'team';

/**
 * Organization represents a workspace context.
 * All resources (workflows, integrations, etc.) are scoped to an organization.
 */
export interface Organization {
  id: number;
  name: string;
  slug: string;
  type: OrganizationType;
  role: OrganizationRole;
  isOwner: boolean;
  memberCount?: number;
  createdAt?: string;
  updatedAt?: string;
}

// =============================================================================
// Member Types
// =============================================================================

/**
 * Member status tracks the lifecycle of an organization membership.
 */
export type MemberStatus = 'pending' | 'active' | 'suspended';

/**
 * Member represents a user's membership in an organization.
 */
export interface Member {
  id: number;
  userId: number;
  email: string;
  firstName?: string;
  lastName?: string;
  role: OrganizationRole;
  status: MemberStatus;
  joinedAt?: string;
  avatarUrl?: string;
}

// =============================================================================
// Invitation Types
// =============================================================================

/**
 * Invitation status tracks the lifecycle of a team invitation.
 */
export type InvitationStatus = 'pending' | 'accepted' | 'expired' | 'revoked';

/**
 * Invitation represents a pending or processed team invitation.
 */
export interface Invitation {
  id: number;
  email: string;
  role: OrganizationRole;
  status: InvitationStatus;
  expiresAt: string;
  invitedBy: string;
  createdAt: string;
  organizationName?: string;
}

// =============================================================================
// API Response Types
// =============================================================================

/**
 * Response from GET /api/organizations
 */
export interface OrganizationsResponse {
  organizations: Organization[];
  current_organization_id: number | null;
}

/**
 * Response from POST /api/organizations (create team)
 */
export interface CreateOrganizationResponse {
  organization: Organization;
}

/**
 * Response from POST /api/organizations/:id/switch
 */
export interface SwitchOrganizationResponse {
  organization: Organization;
  role: OrganizationRole;
  message: string;
}

/**
 * Response from GET /api/organizations/:id/members
 */
export interface MembersResponse {
  members: Member[];
}

/**
 * Response from GET /api/organizations/:id/invitations
 */
export interface InvitationsResponse {
  invitations: Invitation[];
}

/**
 * Response from GET /api/invitations/:token (invitation details for accept page)
 */
export interface InvitationDetailsResponse {
  invitation: Invitation;
  organizationName: string;
  inviterName: string;
}

/**
 * Response from POST /api/invitations/:token/accept
 */
export interface AcceptInvitationResponse {
  success: boolean;
  organization: Organization;
}

// =============================================================================
// Request Types
// =============================================================================

/**
 * Request body for creating a new team organization.
 */
export interface CreateOrganizationRequest {
  name: string;
}

/**
 * Request body for inviting a member to an organization.
 */
export interface InviteMemberRequest {
  email: string;
  role: OrganizationRole;
}

/**
 * Request body for updating a member's role.
 */
export interface UpdateMemberRoleRequest {
  role: OrganizationRole;
}

// =============================================================================
// Permission Helpers
// =============================================================================

/**
 * Role hierarchy for permission checks.
 * Higher index = more permissions.
 */
export const ROLE_HIERARCHY: OrganizationRole[] = ['consultant', 'user', 'admin', 'owner'];

/**
 * Get role display name for UI.
 */
export function getRoleDisplayName(role: OrganizationRole): string {
  const names: Record<OrganizationRole, string> = {
    owner: 'Owner',
    admin: 'Admin',
    user: 'User',
    consultant: 'Consultant',
  };
  return names[role];
}

/**
 * Get role description for UI (e.g., in role selector).
 */
export function getRoleDescription(role: OrganizationRole): string {
  const descriptions: Record<OrganizationRole, string> = {
    owner: 'Full control including billing and deletion',
    admin: 'Can manage members and all workflows',
    user: 'Can create and manage own workflows',
    consultant: 'External access, workflows need approval',
  };
  return descriptions[role];
}

/**
 * Check if a role can invite members.
 */
export function canInviteMembers(role: OrganizationRole | null): boolean {
  return role !== null && ['owner', 'admin', 'consultant'].includes(role);
}

/**
 * Check if a role can manage members (change roles, remove).
 */
export function canManageMembers(role: OrganizationRole | null): boolean {
  return role !== null && ['owner', 'admin'].includes(role);
}

/**
 * Check if a role can manage billing.
 */
export function canManageBilling(role: OrganizationRole | null): boolean {
  return role === 'owner';
}

/**
 * Check if a role can delete the organization.
 */
export function canDeleteOrganization(role: OrganizationRole | null): boolean {
  return role === 'owner';
}

/**
 * Get roles that a given role can assign to others.
 * - Consultants can only invite users
 * - Admins can invite admins, users, and consultants
 * - Owners can invite all roles except owner
 */
export function getAssignableRoles(role: OrganizationRole | null): OrganizationRole[] {
  switch (role) {
    case 'owner':
      return ['admin', 'user', 'consultant'];
    case 'admin':
      return ['admin', 'user', 'consultant'];
    case 'consultant':
      return ['user'];
    default:
      return [];
  }
}

// =============================================================================
// Workflow Transfer Types
// =============================================================================

/**
 * Request body for transferring workflows to a team.
 * Users explicitly select which workflows to transfer (one-way operation).
 */
export interface TransferWorkflowsRequest {
  workflowIds: string[];
}

/**
 * Response from workflow transfer operation.
 */
export interface TransferWorkflowsResponse {
  success: boolean;
  transferredCount: number;
  workflowIds: string[];
}

// =============================================================================
// Workflow Approval Types (for Consultants)
// =============================================================================

/**
 * Workflow approval status for consultant-created workflows.
 */
export type ApprovalStatus = 'pending' | 'approved' | 'rejected' | 'draft';

/**
 * Workflow approval request tracking.
 */
export interface WorkflowApproval {
  id: number;
  workflowId: string;
  workflowName: string;
  organizationId: number;
  requestedBy: {
    id: number;
    email: string;
    name: string;
  };
  requestedAt: string;
  status: ApprovalStatus;
  reviewedBy?: {
    id: number;
    email: string;
    name: string;
  };
  reviewedAt?: string;
  reviewNotes?: string;
}

/**
 * Response from GET /api/organizations/:id/approvals
 */
export interface ApprovalsResponse {
  approvals: WorkflowApproval[];
}

/**
 * Response from POST /api/workflows/:id/request-approval
 */
export interface RequestApprovalResponse {
  success: boolean;
  approval: WorkflowApproval;
}

/**
 * Request body for reviewing an approval.
 */
export interface ReviewApprovalRequest {
  status: 'approved' | 'rejected';
  notes?: string;
}

/**
 * Response from POST /api/approvals/:id/review
 */
export interface ReviewApprovalResponse {
  success: boolean;
  approval: WorkflowApproval;
}

// =============================================================================
// Shared Integration Types
// =============================================================================

/**
 * An OAuth connection that can be shared with the team.
 */
export interface SharedIntegration {
  id: string;
  provider: string;
  integrationType: string;
  accountEmail?: string;
  accountName?: string;
  userId: number;
  sharedWithOrg: boolean;
  sharedAt?: string;
  sharedByName?: string;
}

/**
 * Response from GET /api/organizations/:id/integrations
 */
export interface SharedIntegrationsResponse {
  integrations: SharedIntegration[];
}

/**
 * Check if a role can review workflow approvals.
 */
export function canReviewApprovals(role: OrganizationRole | null): boolean {
  return role !== null && ['owner', 'admin'].includes(role);
}
