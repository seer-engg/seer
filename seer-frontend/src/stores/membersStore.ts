/**
 * Members Store
 *
 * Manages team members and invitations for the current organization.
 * Integrates with the organization store to know which org is active.
 */

import { create } from 'zustand';
import { organizationApi } from '@/lib/organization-api';
import type { Member, Invitation, OrganizationRole } from '@/types/organization';
import { useOrganizationStore } from './organizationStore';

// =============================================================================
// Types
// =============================================================================

export interface MembersState {
  // Data
  members: Member[];
  invitations: Invitation[];

  // Loading states
  isLoading: boolean;
  isInviting: boolean;
  error: string | null;

  // Actions
  fetchMembers: (orgId?: number) => Promise<void>;
  fetchInvitations: (orgId?: number) => Promise<void>;
  inviteMember: (email: string, role: OrganizationRole) => Promise<void>;
  updateMemberRole: (userId: number, role: OrganizationRole) => Promise<void>;
  removeMember: (userId: number) => Promise<void>;
  revokeInvitation: (invitationId: number) => Promise<void>;
  resendInvitation: (invitationId: number) => Promise<void>;

  // Reset
  reset: () => void;
}

// =============================================================================
// Initial State
// =============================================================================

const initialState = {
  members: [],
  invitations: [],
  isLoading: false,
  isInviting: false,
  error: null,
};

// =============================================================================
// Helper to get current org ID
// =============================================================================

const getCurrentOrgId = (): number | null => {
  return useOrganizationStore.getState().currentOrganization?.id ?? null;
};

// =============================================================================
// Store
// =============================================================================

/* eslint-disable max-lines-per-function */
export const useMembersStore = create<MembersState>((set, get) => ({
  ...initialState,

  // ---------------------------------------------------------------------------
  // Fetch members for the current organization
  // ---------------------------------------------------------------------------
  fetchMembers: async (orgId?: number) => {
    const targetOrgId = orgId ?? getCurrentOrgId();
    if (!targetOrgId) {
      set({ members: [], error: 'No organization selected' });
      return;
    }

    set({ isLoading: true, error: null });
    try {
      const members = await organizationApi.getMembers(targetOrgId);
      set({ members, isLoading: false });
    } catch (error) {
      console.error('Failed to fetch members:', error);
      set({ error: 'Failed to fetch members', isLoading: false });
    }
  },

  // ---------------------------------------------------------------------------
  // Fetch invitations for the current organization
  // ---------------------------------------------------------------------------
  fetchInvitations: async (orgId?: number) => {
    const targetOrgId = orgId ?? getCurrentOrgId();
    if (!targetOrgId) {
      set({ invitations: [] });
      return;
    }

    try {
      const invitations = await organizationApi.getInvitations(targetOrgId);
      set({ invitations });
    } catch (error) {
      console.error('Failed to fetch invitations:', error);
      // Don't set error state for invitations - they're secondary data
    }
  },

  // ---------------------------------------------------------------------------
  // Invite a new member
  // ---------------------------------------------------------------------------
  inviteMember: async (email, role) => {
    const orgId = getCurrentOrgId();
    if (!orgId) {
      throw new Error('No organization selected');
    }

    set({ isInviting: true, error: null });
    try {
      await organizationApi.inviteMember(orgId, email, role);
      // Refresh invitations list
      await get().fetchInvitations(orgId);
      set({ isInviting: false });
    } catch (error) {
      console.error('Failed to invite member:', error);
      set({ error: 'Failed to send invitation', isInviting: false });
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Update a member's role
  // ---------------------------------------------------------------------------
  updateMemberRole: async (userId, role) => {
    const orgId = getCurrentOrgId();
    if (!orgId) {
      throw new Error('No organization selected');
    }

    try {
      await organizationApi.updateMemberRole(orgId, userId, role);
      // Update local state
      set((state) => ({
        members: state.members.map((m) =>
          m.userId === userId ? { ...m, role } : m
        ),
      }));
    } catch (error) {
      console.error('Failed to update member role:', error);
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Remove a member from the organization
  // ---------------------------------------------------------------------------
  removeMember: async (userId) => {
    const orgId = getCurrentOrgId();
    if (!orgId) {
      throw new Error('No organization selected');
    }

    try {
      await organizationApi.removeMember(orgId, userId);
      // Update local state
      set((state) => ({
        members: state.members.filter((m) => m.userId !== userId),
      }));
    } catch (error) {
      console.error('Failed to remove member:', error);
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Revoke an invitation
  // ---------------------------------------------------------------------------
  revokeInvitation: async (invitationId) => {
    const orgId = getCurrentOrgId();
    if (!orgId) {
      throw new Error('No organization selected');
    }

    try {
      await organizationApi.revokeInvitation(orgId, invitationId);
      // Update local state
      set((state) => ({
        invitations: state.invitations.filter((i) => i.id !== invitationId),
      }));
    } catch (error) {
      console.error('Failed to revoke invitation:', error);
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Resend an invitation
  // ---------------------------------------------------------------------------
  resendInvitation: async (invitationId) => {
    const orgId = getCurrentOrgId();
    if (!orgId) {
      throw new Error('No organization selected');
    }

    try {
      await organizationApi.resendInvitation(orgId, invitationId);
    } catch (error) {
      console.error('Failed to resend invitation:', error);
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Reset store to initial state
  // ---------------------------------------------------------------------------
  reset: () => {
    set(initialState);
  },
}));
