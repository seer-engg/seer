/**
 * Zustand store for browser session state and profile cache.
 *
 * Manages:
 * - Active streaming session state
 * - Browser profiles cache
 * - Connection status
 */
import type { BrowserProfile, StreamStatus } from '@/types/browser';
import { createStore } from './createStore';

export interface BrowserSessionState {
  sessionId: string | null;
  profileId: string | null;
  wsUrl: string | null;
  recordingId: string | null;
  status: StreamStatus;
  currentUrl: string;
  isLoading: boolean;
  error: string | null;
}

interface BrowserStore {
  // Session state
  session: BrowserSessionState;

  // Profile cache (synced with React Query)
  profiles: BrowserProfile[];
  profilesLoading: boolean;

  // Session actions
  setSession: (session: Partial<BrowserSessionState>) => void;
  clearSession: () => void;
  setSessionStatus: (status: StreamStatus) => void;
  setSessionUrl: (url: string) => void;
  setSessionError: (error: string | null) => void;
  setSessionLoading: (loading: boolean) => void;

  // Profile actions
  setProfiles: (profiles: BrowserProfile[]) => void;
  setProfilesLoading: (loading: boolean) => void;
  addProfile: (profile: BrowserProfile) => void;
  removeProfile: (profileId: string) => void;
  updateProfile: (profileId: string, updates: Partial<BrowserProfile>) => void;
}

const initialSessionState: BrowserSessionState = {
  sessionId: null,
  profileId: null,
  wsUrl: null,
  recordingId: null,
  status: 'closed',
  currentUrl: '',
  isLoading: false,
  error: null,
};

export const useBrowserStore = createStore<BrowserStore>((set) => ({
  // Initial state
  session: { ...initialSessionState },
  profiles: [],
  profilesLoading: false,

  // Session actions
  setSession: (updates) =>
    set((state) => ({
      session: { ...state.session, ...updates },
    })),

  clearSession: () =>
    set({
      session: { ...initialSessionState },
    }),

  setSessionStatus: (status) =>
    set((state) => ({
      session: { ...state.session, status },
    })),

  setSessionUrl: (url) =>
    set((state) => ({
      session: { ...state.session, currentUrl: url },
    })),

  setSessionError: (error) =>
    set((state) => ({
      session: { ...state.session, error, status: error ? 'error' : state.session.status },
    })),

  setSessionLoading: (isLoading) =>
    set((state) => ({
      session: { ...state.session, isLoading },
    })),

  // Profile actions
  setProfiles: (profiles) => set({ profiles }),

  setProfilesLoading: (profilesLoading) => set({ profilesLoading }),

  addProfile: (profile) =>
    set((state) => ({
      profiles: [...state.profiles, profile],
    })),

  removeProfile: (profileId) =>
    set((state) => ({
      profiles: state.profiles.filter((p) => p.id !== profileId),
    })),

  updateProfile: (profileId, updates) =>
    set((state) => ({
      profiles: state.profiles.map((p) =>
        p.id === profileId ? { ...p, ...updates } : p
      ),
    })),
}));
