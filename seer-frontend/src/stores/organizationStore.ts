/**
 * Organization Store
 *
 * Manages organization state including the current active organization,
 * list of user's organizations, and organization switching.
 *
 * Key design decisions:
 * - Uses backend current_organization_id as the source of truth
 * - Persists the last selected org locally as a convenience fallback only
 * - Provides permission helper functions (canManageBilling, canInviteMembers, etc.)
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { organizationApi } from '@/lib/organization-api';
import { invalidateBuilderCatalogQueries, queryClient } from '@/lib/query-client';
import { workflowKeys } from '@/lib/query-keys';
import type { Organization, OrganizationRole } from '@/types/organization';
import {
  canInviteMembers as checkCanInvite,
  canManageMembers as checkCanManage,
  canManageBilling as checkCanBilling,
  canDeleteOrganization as checkCanDelete,
} from '@/types/organization';
import { useCanvasStore } from './canvasStore';
import { useChatStore } from './chatStore';
import { useTriggersStore } from './triggersStore';
import { useUIStore } from './uiStore';
import { useWorkflowCollaborationStore } from './workflowCollaborationStore';
import { useWorkflowStore } from './workflowStore';

// =============================================================================
export interface OrganizationState {
  // Data
  organizations: Organization[];
  currentOrganization: Organization | null;

  // Loading states
  isLoading: boolean;
  isInitialized: boolean;
  isSwitching: boolean;
  error: string | null;

  // Actions
  fetchOrganizations: () => Promise<void>;
  setCurrentOrganization: (org: Organization) => void;
  switchOrganization: (orgId: number) => Promise<void>;
  createOrganization: (name: string) => Promise<Organization>;
  updateOrganization: (orgId: number, updates: { name?: string }) => Promise<void>;
  deleteOrganization: (orgId: number) => Promise<void>;

  // Helpers
  getPersonalOrg: () => Organization | undefined;
  isCurrentOrgPersonal: () => boolean;
  getCurrentRole: () => OrganizationRole | null;
  canManageBilling: () => boolean;
  canInviteMembers: () => boolean;
  canManageMembers: () => boolean;
  canDeleteOrg: () => boolean;

  // Reset
  reset: () => void;
}

// =============================================================================
// Initial State
// =============================================================================

const initialState = {
  organizations: [],
  currentOrganization: null,
  isLoading: false,
  isInitialized: false,
  isSwitching: false,
  error: null,
};

function resetWorkflowBuilderLocalState() {
  useCanvasStore.getState().reset();
  useTriggersStore.setState({ workflowTriggers: new Map() });
  useUIStore.getState().resetUIState();
  useChatStore.getState().resetChatState();
  useWorkflowCollaborationStore.getState().clearAll();
  useWorkflowStore.setState({
    error: null,
    isCreating: false,
    isUpdating: false,
    isSavingDraft: false,
    isPublishing: false,
    isDeleting: false,
    isRestoringVersion: false,
    isExecuting: false,
    selectedWorkflowId: null,
    workflowName: 'My Workflow',
    workflowInputData: {},
    isLoadingWorkflow: false,
  });
}

// =============================================================================
// Store
// =============================================================================

/* eslint-disable max-lines-per-function */
export const useOrganizationStore = create<OrganizationState>()(
  persist(
    (set, get) => ({
      ...initialState,

      // -----------------------------------------------------------------------
      // Fetch all organizations for the current user
      // -----------------------------------------------------------------------
      fetchOrganizations: async () => {
        set({ isLoading: true, error: null });
        try {
          const response = await organizationApi.getOrganizations();
          const orgs = response.organizations;
          const current = get().currentOrganization;
          const serverSelectedOrg =
            response.current_organization_id != null
              ? orgs.find((o) => o.id === response.current_organization_id)
              : undefined;

          // Prefer the server-selected org. Fall back to the locally persisted
          // org only if the backend does not provide one.
          let selectedOrg =
            serverSelectedOrg ??
            (current ? orgs.find((o) => o.id === current.id) : undefined) ??
            orgs.find((o) => o.type === 'personal');

          // Fallback to first org if personal not found
          if (!selectedOrg && orgs.length > 0) {
            selectedOrg = orgs[0];
          }

          set({
            organizations: orgs,
            currentOrganization: selectedOrg || null,
            isLoading: false,
            isInitialized: true,
          });
        } catch (error) {
          console.error('Failed to fetch organizations:', error);
          set({
            error: 'Failed to fetch organizations',
            isLoading: false,
            isInitialized: true,
          });
        }
      },

      // -----------------------------------------------------------------------
      // Set current organization (local state only, no API call)
      // -----------------------------------------------------------------------
      setCurrentOrganization: (org) => {
        set({ currentOrganization: org });
      },

      // -----------------------------------------------------------------------
      // Switch organization context
      // -----------------------------------------------------------------------
      switchOrganization: async (orgId) => {
        const org = get().organizations.find((o) => o.id === orgId);
        if (!org) {
          throw new Error('Organization not found');
        }

        // Don't switch if already on this org
        if (get().currentOrganization?.id === orgId) {
          return;
        }

        set({ isSwitching: true, error: null });

        try {
          const response = await organizationApi.switchOrganization(orgId);
          const switchedOrg = {
            ...org,
            ...response.organization,
            role: response.role,
          };

          queryClient.removeQueries({ queryKey: workflowKeys.all });
          resetWorkflowBuilderLocalState();

          set((state) => ({
            organizations: state.organizations.map((item) =>
              item.id === switchedOrg.id ? switchedOrg : item
            ),
            currentOrganization: switchedOrg,
            isSwitching: false,
          }));

          await invalidateBuilderCatalogQueries(queryClient);

          const { useUsageStore } = await import('./usageStore');
          const { useSubscriptionStore } = await import('./subscriptionStore');

          // Fire and forget - these will update in the background
          Promise.all([
            useUsageStore.getState().fetchUsage(),
            useSubscriptionStore.getState().fetchSubscription(),
          ]).catch(console.error);
        } catch (error) {
          console.error('Failed to switch organization:', error);
          set({
            error: 'Failed to switch organization',
            isSwitching: false,
          });
          throw error;
        }
      },

      // -----------------------------------------------------------------------
      // Create a new team organization
      // -----------------------------------------------------------------------
      createOrganization: async (name) => {
        set({ isLoading: true, error: null });
        try {
          const newOrg = await organizationApi.createOrganization(name);

          if (!newOrg?.id) {
            throw new Error('Organization creation failed: invalid response from server');
          }

          set((state) => ({
            organizations: [...state.organizations, newOrg],
            isLoading: false,
          }));

          // Auto-switch to the new organization
          await get().switchOrganization(newOrg.id);

          return newOrg;
        } catch (error) {
          console.error('Failed to create organization:', error);
          set({
            error: 'Failed to create organization',
            isLoading: false,
          });
          throw error;
        }
      },

      // -----------------------------------------------------------------------
      // Update organization details
      // -----------------------------------------------------------------------
      updateOrganization: async (orgId, updates) => {
        try {
          const updatedOrg = await organizationApi.updateOrganization(orgId, updates);

          set((state) => ({
            organizations: state.organizations.map((o) =>
              o.id === orgId ? { ...o, ...updatedOrg } : o
            ),
            currentOrganization:
              state.currentOrganization?.id === orgId
                ? { ...state.currentOrganization, ...updatedOrg }
                : state.currentOrganization,
          }));
        } catch (error) {
          console.error('Failed to update organization:', error);
          throw error;
        }
      },

      // -----------------------------------------------------------------------
      // Delete an organization
      // -----------------------------------------------------------------------
      deleteOrganization: async (orgId) => {
        const current = get().currentOrganization;
        const personalOrg = get().getPersonalOrg();

        try {
          await organizationApi.deleteOrganization(orgId);

          set((state) => ({
            organizations: state.organizations.filter((o) => o.id !== orgId),
          }));

          // If we deleted the current org, switch to personal
          if (current?.id === orgId && personalOrg) {
            await get().switchOrganization(personalOrg.id);
          }
        } catch (error) {
          console.error('Failed to delete organization:', error);
          throw error;
        }
      },

      // -----------------------------------------------------------------------
      // Helper: Get personal organization
      // -----------------------------------------------------------------------
      getPersonalOrg: () => {
        return get().organizations.find((o) => o.type === 'personal');
      },

      // -----------------------------------------------------------------------
      // Helper: Check if current org is personal
      // -----------------------------------------------------------------------
      isCurrentOrgPersonal: () => {
        return get().currentOrganization?.type === 'personal';
      },

      // -----------------------------------------------------------------------
      // Helper: Get current user's role in the current org
      // -----------------------------------------------------------------------
      getCurrentRole: () => {
        return get().currentOrganization?.role || null;
      },

      // -----------------------------------------------------------------------
      // Permission helpers
      // -----------------------------------------------------------------------
      canManageBilling: () => {
        return checkCanBilling(get().currentOrganization?.role || null);
      },

      canInviteMembers: () => {
        return checkCanInvite(get().currentOrganization?.role || null);
      },

      canManageMembers: () => {
        return checkCanManage(get().currentOrganization?.role || null);
      },

      canDeleteOrg: () => {
        return checkCanDelete(get().currentOrganization?.role || null);
      },

      // -----------------------------------------------------------------------
      // Reset store to initial state
      // -----------------------------------------------------------------------
      reset: () => {
        useWorkflowCollaborationStore.getState().clearAll();
        set(initialState);
      },
    }),
    {
      name: 'seer-organization',
      // Only persist the current organization selection
      partialize: (state) => ({
        currentOrganization: state.currentOrganization,
      }),
    }
  )
);
