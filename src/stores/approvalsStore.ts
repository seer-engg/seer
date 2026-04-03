/**
 * Approvals Store
 *
 * Manages workflow approval requests for consultant-created workflows.
 * Integrates with the organization store to know which org is active.
 */

import { create } from 'zustand';
import { organizationApi } from '@/lib/organization-api';
import type { WorkflowApproval } from '@/types/organization';
import { useOrganizationStore } from './organizationStore';

// =============================================================================
// Types
// =============================================================================

export interface ApprovalsState {
  // Data
  approvals: WorkflowApproval[];

  // Loading states
  isLoading: boolean;
  isReviewing: boolean;
  error: string | null;

  // Actions
  fetchApprovals: (orgId?: number) => Promise<void>;
  requestApproval: (workflowId: string) => Promise<void>;
  reviewApproval: (
    approvalId: number,
    status: 'approved' | 'rejected',
    notes?: string
  ) => Promise<void>;

  // Helpers
  getPendingApprovals: () => WorkflowApproval[];
  getApprovalForWorkflow: (workflowId: string) => WorkflowApproval | null;

  // Reset
  reset: () => void;
}

// =============================================================================
// Initial State
// =============================================================================

const initialState = {
  approvals: [],
  isLoading: false,
  isReviewing: false,
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

export const useApprovalsStore = create<ApprovalsState>((set, get) => ({
  ...initialState,

  // ---------------------------------------------------------------------------
  // Fetch approvals for the current organization
  // ---------------------------------------------------------------------------
  fetchApprovals: async (orgId?: number) => {
    const targetOrgId = orgId ?? getCurrentOrgId();
    if (!targetOrgId) {
      set({ approvals: [], error: 'No organization selected' });
      return;
    }

    set({ isLoading: true, error: null });
    try {
      const approvals = await organizationApi.getApprovals(targetOrgId);
      set({ approvals, isLoading: false });
    } catch (error) {
      console.error('Failed to fetch approvals:', error);
      set({ error: 'Failed to fetch approvals', isLoading: false });
    }
  },

  // ---------------------------------------------------------------------------
  // Request approval for a workflow (used by consultants)
  // ---------------------------------------------------------------------------
  requestApproval: async (workflowId) => {
    set({ error: null });
    try {
      const response = await organizationApi.requestApproval(workflowId);
      // Add the new approval to the list
      set((state) => ({
        approvals: [...state.approvals, response.approval],
      }));
    } catch (error) {
      console.error('Failed to request approval:', error);
      set({ error: 'Failed to request approval' });
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Review an approval (approve or reject)
  // ---------------------------------------------------------------------------
  reviewApproval: async (approvalId, status, notes) => {
    set({ isReviewing: true, error: null });
    try {
      const response = await organizationApi.reviewApproval(approvalId, status, notes);
      // Update the approval in the list
      set((state) => ({
        approvals: state.approvals.map((a) =>
          a.id === approvalId ? response.approval : a
        ),
        isReviewing: false,
      }));
    } catch (error) {
      console.error('Failed to review approval:', error);
      set({ error: 'Failed to review approval', isReviewing: false });
      throw error;
    }
  },

  // ---------------------------------------------------------------------------
  // Get pending approvals only
  // ---------------------------------------------------------------------------
  getPendingApprovals: () => {
    return get().approvals.filter((a) => a.status === 'pending');
  },

  // ---------------------------------------------------------------------------
  // Get approval for a specific workflow
  // ---------------------------------------------------------------------------
  getApprovalForWorkflow: (workflowId) => {
    return get().approvals.find((a) => a.workflowId === workflowId) ?? null;
  },

  // ---------------------------------------------------------------------------
  // Reset store to initial state
  // ---------------------------------------------------------------------------
  reset: () => {
    set(initialState);
  },
}));
