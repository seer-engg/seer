/**
 * Organization API Client
 *
 * Handles all API calls related to organizations, members, and invitations.
 * Uses the backendApiClient for authenticated requests.
 */

import { backendApiClient } from './api-client';
import type {
  Organization,
  OrganizationsResponse,
  CreateOrganizationResponse,
  SwitchOrganizationResponse,
  Member,
  MembersResponse,
  Invitation,
  InvitationsResponse,
  InvitationDetailsResponse,
  AcceptInvitationResponse,
  OrganizationRole,
  TransferWorkflowsRequest,
  TransferWorkflowsResponse,
  WorkflowApproval,
  ApprovalsResponse,
  RequestApprovalResponse,
  ReviewApprovalRequest,
  ReviewApprovalResponse,
  SharedIntegration,
  SharedIntegrationsResponse,
} from '@/types/organization';

// =============================================================================
// Organization APIs
// =============================================================================

/**
 * Fetch all organizations the current user is a member of.
 * Returns both personal workspace and team organizations.
 */
export async function getOrganizations(): Promise<OrganizationsResponse> {
  return backendApiClient.request<OrganizationsResponse>('/api/organizations');
}

/**
 * Get a specific organization by ID.
 */
export async function getOrganization(orgId: number): Promise<Organization> {
  return backendApiClient.request<Organization>(`/api/organizations/${orgId}`);
}

/**
 * Create a new team organization.
 * Requires an active subscription - the subscription is transferred to the new team.
 */
export async function createOrganization(name: string): Promise<Organization> {
  return backendApiClient.request<Organization>('/api/organizations', {
    method: 'POST',
    body: { name, transfer_subscription: true },
  });
}

/**
 * Switch the active organization context.
 * Persists the new active organization on the backend.
 */
export async function switchOrganization(orgId: number): Promise<SwitchOrganizationResponse> {
  return backendApiClient.request<SwitchOrganizationResponse>(
    `/api/organizations/${orgId}/switch`,
    { method: 'POST' }
  );
}

/**
 * Update organization details (name, etc.).
 * Only owners can update organization details.
 */
export async function updateOrganization(
  orgId: number,
  updates: { name?: string }
): Promise<Organization> {
  return backendApiClient.request<Organization>(`/api/organizations/${orgId}`, {
    method: 'PATCH',
    body: updates,
  });
}

/**
 * Delete an organization.
 * Only owners can delete organizations.
 * Personal organizations cannot be deleted.
 */
export async function deleteOrganization(orgId: number): Promise<void> {
  await backendApiClient.request(`/api/organizations/${orgId}`, {
    method: 'DELETE',
  });
}

// =============================================================================
// Member APIs
// =============================================================================

/**
 * Fetch all members of an organization.
 */
export async function getMembers(orgId: number): Promise<Member[]> {
  const response = await backendApiClient.request<MembersResponse>(
    `/api/organizations/${orgId}/members`
  );
  return response.members;
}

/**
 * Update a member's role within the organization.
 */
export async function updateMemberRole(
  orgId: number,
  userId: number,
  role: OrganizationRole
): Promise<Member> {
  return backendApiClient.request<Member>(
    `/api/organizations/${orgId}/members/${userId}`,
    {
      method: 'PATCH',
      body: { role },
    }
  );
}

/**
 * Remove a member from the organization.
 */
export async function removeMember(orgId: number, userId: number): Promise<void> {
  await backendApiClient.request(`/api/organizations/${orgId}/members/${userId}`, {
    method: 'DELETE',
  });
}

/**
 * Leave an organization (remove self).
 * Owners cannot leave - they must transfer ownership first.
 */
export async function leaveOrganization(orgId: number): Promise<void> {
  await backendApiClient.request(`/api/organizations/${orgId}/leave`, {
    method: 'POST',
  });
}

// =============================================================================
// Invitation APIs
// =============================================================================

/**
 * Fetch all pending and recent invitations for an organization.
 */
export async function getInvitations(orgId: number): Promise<Invitation[]> {
  const response = await backendApiClient.request<InvitationsResponse>(
    `/api/organizations/${orgId}/invitations`
  );
  return response.invitations;
}

/**
 * Send an invitation to join the organization.
 */
export async function inviteMember(
  orgId: number,
  email: string,
  role: OrganizationRole
): Promise<Invitation> {
  return backendApiClient.request<Invitation>(
    `/api/organizations/${orgId}/invitations`,
    {
      method: 'POST',
      body: { email, role },
    }
  );
}

/**
 * Revoke a pending invitation.
 */
export async function revokeInvitation(orgId: number, invitationId: number): Promise<void> {
  await backendApiClient.request(
    `/api/organizations/${orgId}/invitations/${invitationId}`,
    { method: 'DELETE' }
  );
}

/**
 * Resend an invitation email.
 */
export async function resendInvitation(orgId: number, invitationId: number): Promise<void> {
  await backendApiClient.request(
    `/api/organizations/${orgId}/invitations/${invitationId}/resend`,
    { method: 'POST' }
  );
}

/**
 * Get invitation details by token (for the invitation accept page).
 * This is a public endpoint that doesn't require authentication.
 */
export async function getInvitationByToken(token: string): Promise<InvitationDetailsResponse> {
  return backendApiClient.request<InvitationDetailsResponse>(`/api/organizations/invitations/${token}`);
}

/**
 * Accept an invitation using the token.
 * Requires authentication - the invitation is linked to the authenticated user.
 */
export async function acceptInvitation(token: string): Promise<AcceptInvitationResponse> {
  return backendApiClient.request<AcceptInvitationResponse>(
    `/api/organizations/invitations/${token}/accept`,
    { method: 'POST' }
  );
}

/**
 * Decline an invitation using the token.
 */
export async function declineInvitation(token: string): Promise<void> {
  await backendApiClient.request(`/api/invitations/${token}/decline`, {
    method: 'POST',
  });
}

// =============================================================================
// Workflow Transfer APIs
// =============================================================================

/**
 * Transfer workflows from personal workspace to a team organization.
 * This is a one-way operation and cannot be undone.
 */
export async function transferWorkflows(
  orgId: number,
  workflowIds: string[]
): Promise<TransferWorkflowsResponse> {
  return backendApiClient.request<TransferWorkflowsResponse>(
    `/api/organizations/${orgId}/transfer-workflows`,
    {
      method: 'POST',
      body: { workflowIds },
    }
  );
}

// =============================================================================
// Workflow Approval APIs (for Consultants)
// =============================================================================

/**
 * Fetch all pending workflow approvals for an organization.
 * Only accessible by owners and admins.
 */
export async function getApprovals(orgId: number): Promise<WorkflowApproval[]> {
  const response = await backendApiClient.request<ApprovalsResponse>(
    `/api/organizations/${orgId}/approvals`
  );
  return response.approvals;
}

/**
 * Request approval for a workflow (used by consultants).
 */
export async function requestApproval(workflowId: string): Promise<RequestApprovalResponse> {
  return backendApiClient.request<RequestApprovalResponse>(
    `/api/workflows/${workflowId}/request-approval`,
    { method: 'POST' }
  );
}

/**
 * Review a workflow approval request (approve or reject).
 * Only accessible by owners and admins.
 */
export async function reviewApproval(
  approvalId: number,
  status: 'approved' | 'rejected',
  notes?: string
): Promise<ReviewApprovalResponse> {
  return backendApiClient.request<ReviewApprovalResponse>(
    `/api/approvals/${approvalId}/review`,
    {
      method: 'POST',
      body: { status, notes },
    }
  );
}

// =============================================================================
// Shared Integration APIs
// =============================================================================

/**
 * Fetch all shared integrations for an organization.
 */
export async function getSharedIntegrations(orgId: number): Promise<SharedIntegration[]> {
  const response = await backendApiClient.request<SharedIntegrationsResponse>(
    `/api/organizations/${orgId}/integrations`
  );
  return response.integrations;
}

/**
 * Share an OAuth connection with the team.
 */
export async function shareIntegration(
  orgId: number,
  connectionId: string
): Promise<SharedIntegration> {
  // Extract numeric ID from "provider:id" format (e.g., "google:1" -> "1")
  const numericId = connectionId.includes(':') ? connectionId.split(':')[1] : connectionId;

  return backendApiClient.request<SharedIntegration>(
    `/api/organizations/${orgId}/integrations/${numericId}/share`,
    { method: 'POST' }
  );
}

/**
 * Unshare an OAuth connection from the team.
 * Only the owner of the connection can unshare it.
 */
export async function unshareIntegration(
  orgId: number,
  connectionId: string
): Promise<void> {
  // Extract numeric ID from "provider:id" format (e.g., "google:1" -> "1")
  const numericId = connectionId.includes(':') ? connectionId.split(':')[1] : connectionId;

  await backendApiClient.request(
    `/api/organizations/${orgId}/integrations/${numericId}/share`,
    { method: 'DELETE' }
  );
}

// =============================================================================
// Export as namespace for consistent API access
// =============================================================================

export const organizationApi = {
  // Organizations
  getOrganizations,
  getOrganization,
  createOrganization,
  switchOrganization,
  updateOrganization,
  deleteOrganization,
  // Members
  getMembers,
  updateMemberRole,
  removeMember,
  leaveOrganization,
  // Invitations
  getInvitations,
  inviteMember,
  revokeInvitation,
  resendInvitation,
  getInvitationByToken,
  acceptInvitation,
  declineInvitation,
  // Workflow Transfer
  transferWorkflows,
  // Approvals
  getApprovals,
  requestApproval,
  reviewApproval,
  // Shared Integrations
  getSharedIntegrations,
  shareIntegration,
  unshareIntegration,
};
