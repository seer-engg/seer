/**
 * Sentry Provider Component
 *
 * Syncs user identification with Clerk authentication.
 * Sets Sentry user context on sign-in and clears on sign-out.
 */
import { useEffect } from 'react';
import { useAuth, useUser } from '@clerk/clerk-react';
import { Sentry } from '@/lib/sentry';

interface SentryProviderProps {
  children: React.ReactNode;
}

export const SentryProvider = ({ children }: SentryProviderProps) => {
  const { isSignedIn, isLoaded } = useAuth();
  const { user } = useUser();

  // Handle user identification
  useEffect(() => {
    // Wait for Clerk to load
    if (!isLoaded) {
      return;
    }

    if (isSignedIn && user?.id) {
      // Set user context in Sentry
      Sentry.setUser({
        id: user.id,
        email: user.primaryEmailAddress?.emailAddress,
        username: user.username || undefined,
      });

      if (import.meta.env.DEV) {
        console.log('[Sentry] User identified:', user.id);
      }
    } else {
      // Clear user context on sign-out
      Sentry.setUser(null);

      if (import.meta.env.DEV && isLoaded) {
        console.log('[Sentry] User signed out, cleared user context');
      }
    }
  }, [isLoaded, isSignedIn, user]);

  return <>{children}</>;
};
