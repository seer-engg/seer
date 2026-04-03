/**
 * Hook for browser profile management with React Query.
 *
 * Provides CRUD operations for browser profiles with optimistic updates.
 */
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';
import {
  listBrowserProfiles,
  createBrowserProfile,
  deleteBrowserProfile,
} from '@/lib/browser-api';
import { browserKeys } from '@/lib/query-keys';
import { useBrowserStore } from '@/stores/browserStore';
import type { BrowserProfile } from '@/types/browser';

export interface UseBrowserProfilesReturn {
  /** List of browser profiles. */
  profiles: BrowserProfile[];
  /** Whether profiles are loading. */
  isLoading: boolean;
  /** Error if profiles failed to load. */
  error: Error | null;
  /** Create a new profile. */
  createProfile: (name: string) => Promise<BrowserProfile>;
  /** Delete a profile. */
  deleteProfile: (profileId: string) => Promise<void>;
  /** Whether a create is in progress. */
  isCreating: boolean;
  /** Whether a delete is in progress. */
  isDeleting: boolean;
  /** Refetch profiles. */
  refetch: () => Promise<void>;
}

export function useBrowserProfiles(): UseBrowserProfilesReturn {
  const queryClient = useQueryClient();
  // Use selectors to prevent re-renders when unrelated store state changes
  const setProfiles = useBrowserStore((state) => state.setProfiles);
  const setProfilesLoading = useBrowserStore((state) => state.setProfilesLoading);
  const addProfile = useBrowserStore((state) => state.addProfile);
  const removeProfile = useBrowserStore((state) => state.removeProfile);

  // Query for profiles
  const {
    data: profiles = [],
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: browserKeys.profiles(),
    queryFn: listBrowserProfiles,
    staleTime: 60 * 1000, // 1 minute
  });

  // Sync to store
  useEffect(() => {
    setProfiles(profiles);
  }, [profiles, setProfiles]);

  useEffect(() => {
    setProfilesLoading(isLoading);
  }, [isLoading, setProfilesLoading]);

  // Create mutation
  const createMutation = useMutation({
    mutationFn: createBrowserProfile,
    onSuccess: (newProfile) => {
      // Optimistic update
      addProfile(newProfile);
      // Invalidate query to refetch
      queryClient.invalidateQueries({ queryKey: browserKeys.profiles() });
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: deleteBrowserProfile,
    onMutate: async (profileId) => {
      // Optimistic removal
      removeProfile(profileId);
    },
    onError: (_, profileId) => {
      // Revert on error by refetching
      queryClient.invalidateQueries({ queryKey: browserKeys.profiles() });
      console.error(`Failed to delete profile ${profileId}`);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: browserKeys.profiles() });
    },
  });

  return {
    profiles,
    isLoading,
    error: error as Error | null,
    createProfile: async (name: string) => {
      return createMutation.mutateAsync(name);
    },
    deleteProfile: async (profileId: string) => {
      await deleteMutation.mutateAsync(profileId);
    },
    isCreating: createMutation.isPending,
    isDeleting: deleteMutation.isPending,
    refetch: async () => {
      await refetch();
    },
  };
}
