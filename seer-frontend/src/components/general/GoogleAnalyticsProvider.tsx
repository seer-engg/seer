/**
 * Google Analytics Provider Component
 * Wraps the app with Google Analytics 4 (GA4) tracking
 * Syncs user identification with Clerk authentication
 */
import { useEffect } from 'react'
import { useAuthStatus, useCurrentUser } from '@/hooks/useAuthProvider'
import { initGoogleAnalytics, setUserId, setUserProperties } from '@/lib/google-analytics'

interface GoogleAnalyticsProviderProps {
  children: React.ReactNode
}

export const GoogleAnalyticsProvider = ({ children }: GoogleAnalyticsProviderProps) => {
  const { isSignedIn, isLoaded } = useAuthStatus()
  const { user } = useCurrentUser()

  // Initialize Google Analytics once on mount
  useEffect(() => {
    const measurementId = initGoogleAnalytics()

    if (import.meta.env.DEV) {
      console.log('[Google Analytics] Provider mounted:', !!measurementId)
    }
  }, [])

  // Handle user identification and sign-out
  useEffect(() => {
    // Wait for Clerk to load before attempting identification
    if (!isLoaded) {
      return
    }

    // Identify user if signed in
    if (isSignedIn && user?.id) {
      setUserId(user.id)
      setUserProperties({
        email: user.primaryEmailAddress?.emailAddress,
        firstName: user.firstName,
        lastName: user.lastName,
        username: user.username,
      })

      if (import.meta.env.DEV) {
        console.log('[Google Analytics] User identified:', user.id)
      }
    }

    // Clear user ID on sign-out
    if (!isSignedIn) {
      setUserId(null)

      if (import.meta.env.DEV) {
        console.log('[Google Analytics] User signed out, cleared user_id')
      }
    }
  }, [isLoaded, isSignedIn, user])

  return <>{children}</>
}
