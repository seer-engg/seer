/**
 * Auth provider abstraction hooks.
 *
 * The auth mode is fetched from the backend at startup (/api/auth/config),
 * so only AUTH_PROVIDER needs to be set on the backend — no VITE_ env var needed.
 *
 * When auth_provider=clerk: delegates to @clerk/clerk-react hooks.
 * When auth_provider=local (default): returns stub values for a single local user.
 *
 * Components should use these hooks instead of importing from @clerk/clerk-react directly.
 */
import { useAuth, useUser, useClerk } from "@clerk/clerk-react";
import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { getBackendBaseUrl } from "@/lib/api-client";

// Runtime auth config — set by initAuthConfig() before React renders
let _authProvider: "clerk" | "local" = "local";

/**
 * Fetch auth config from the backend. Must be called before React renders.
 * Falls back to "local" if the backend is unreachable.
 */
export async function initAuthConfig(): Promise<void> {
  try {
    const baseUrl = getBackendBaseUrl();
    const response = await fetch(`${baseUrl}/api/auth/config`);
    if (response.ok) {
      const data = await response.json();
      _authProvider = data.auth_provider === "clerk" ? "clerk" : "local";
    }
  } catch {
    // Backend unreachable — default to local (self-hosted likely scenario)
    _authProvider = "local";
  }
}

export const getAuthProvider = () => _authProvider;
export const isLocalAuth = () => _authProvider !== "clerk";

const LOCAL_USER_ID = "local-default-user";

const LOCAL_USER_STUB = {
  id: LOCAL_USER_ID,
  primaryEmailAddress: { emailAddress: "local@seer.local" },
  firstName: "Local",
  lastName: "User",
  fullName: "Local User",
  username: "local-user",
  imageUrl: "",
  hasImage: false,
} as const;

/**
 * Drop-in replacement for useAuth() from @clerk/clerk-react.
 */
export function useAuthStatus() {
  if (isLocalAuth()) {
    return {
      isLoaded: true,
      isSignedIn: true as const,
      userId: LOCAL_USER_ID,
    };
  }
  // eslint-disable-next-line react-hooks/rules-of-hooks
  return useAuth();
}

/**
 * Drop-in replacement for useUser() from @clerk/clerk-react.
 */
export function useCurrentUser() {
  if (isLocalAuth()) {
    return {
      user: LOCAL_USER_STUB as any,
      isLoaded: true,
      isSignedIn: true as const,
    };
  }
  // eslint-disable-next-line react-hooks/rules-of-hooks
  return useUser();
}

/**
 * Drop-in replacement for useClerk().signOut().
 * In local mode, just navigates to home.
 */
export function useSignOut() {
  if (isLocalAuth()) {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const navigate = useNavigate();
    // eslint-disable-next-line react-hooks/rules-of-hooks
    return useCallback(() => { navigate("/"); }, [navigate]);
  }
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const { signOut } = useClerk();
  return signOut;
}
