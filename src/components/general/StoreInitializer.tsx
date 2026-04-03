/**
 * StoreInitializer Component
 *
 * Phase 2: Handles initialization of Zustand stores on app mount.
 * This replaces the useEffect initialization logic that was in wrapper hooks.
 *
 * Phase 3 Fix: Wait for Clerk to fully load before initializing stores
 * to ensure bearer token is available for API requests.
 */
import { useEffect } from 'react';
import { useUser } from '@clerk/clerk-react';
import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';
import { useOrganizationStore } from '@/stores/organizationStore';
import { useSubscriptionStore } from '@/stores/subscriptionStore';

export function StoreInitializer() {
  const { user, isLoaded } = useUser();
  const isAuthenticated = isLoaded && !!user;

  // Initialize integration metadata store (requires auth - wait for Clerk)
  useEffect(() => {
    // CRITICAL: Wait for Clerk to load before making authenticated API calls
    if (!isLoaded) return;

    const { loaded, loading, loadMetadata } = useIntegrationMetadataStore.getState();

    if (!loaded && !loading) {
      void loadMetadata().catch(() => undefined);
    }
  }, [isLoaded]);

  // Initialize organization store (requires auth - wait for Clerk)
  useEffect(() => {
    // CRITICAL: Wait for Clerk to load before making authenticated API calls
    if (!isAuthenticated) return;

    const { isInitialized, isLoading, fetchOrganizations } = useOrganizationStore.getState();

    if (!isInitialized && !isLoading) {
      void fetchOrganizations().catch(() => undefined);
    }
  }, [isAuthenticated]);

  // Initialize subscription store (requires auth - wait for Clerk)
  useEffect(() => {
    if (!isAuthenticated) return;

    const { subscription, isLoading, fetchSubscription } = useSubscriptionStore.getState();

    if (!subscription && !isLoading) {
      void fetchSubscription().catch(() => undefined);
    }
  }, [isAuthenticated]);

  return null; // This component doesn't render anything
}
