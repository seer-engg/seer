/**
 * PostHog Provider Component
 * Wraps the app with PostHog analytics initialization
 *
 * Based on official Clerk + PostHog integration guide:
 * https://clerk.com/blog/posthog-events-with-clerk
 */
import { useEffect } from 'react'
import { useAuth, useUser } from '@clerk/clerk-react'
import { posthog, initPostHog } from '@/lib/posthog'

interface PostHogProviderProps {
  children: React.ReactNode
}

export const PostHogProvider = ({ children }: PostHogProviderProps) => {
  const { isSignedIn, isLoaded } = useAuth()
  const { user } = useUser()

  // Initialize PostHog once on mount
  useEffect(() => {
    const ph = initPostHog()

    if (import.meta.env.DEV) {
      console.log('[PostHog] Initialized:', !!ph)
    }
  }, [])

  // Handle user identification and sign-out
  useEffect(() => {
    if (import.meta.env.DEV) {
      console.log('[PostHog Debug]', {
        isLoaded,
        isSignedIn,
        userId: user?.id,
        currentDistinctId: posthog.get_distinct_id(),
      })
    }

    // Wait for Clerk to load before attempting identification
    if (!isLoaded) {
      if (import.meta.env.DEV) {
        console.log('[PostHog] Clerk not loaded yet, waiting...')
      }
      return
    }

    // Identify user if signed in - ALWAYS identify with Clerk ID
    if (isSignedIn && user?.id) {
      const currentId = posthog.get_distinct_id()

      // Only identify if current ID doesn't match Clerk user ID
      if (currentId !== user.id) {
        posthog.identify(user.id, {
          email: user.primaryEmailAddress?.emailAddress,
          firstName: user.firstName,
          lastName: user.lastName,
          username: user.username,
        })

        if (import.meta.env.DEV) {
          console.log('[PostHog] User identified:', user.id, '(was:', currentId, ')')
        }
      } else if (import.meta.env.DEV) {
        console.log('[PostHog] Already identified with correct ID:', currentId)
      }
    }

    // Reset on sign-out
    if (!isSignedIn && posthog.get_distinct_id()) {
      posthog.reset()

      if (import.meta.env.DEV) {
        console.log('[PostHog] User signed out, session reset')
      }
    }
  }, [isLoaded, isSignedIn, user])

  return <>{children}</>
}
