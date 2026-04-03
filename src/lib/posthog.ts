/**
 * PostHog Analytics Client Configuration
 *
 * Based on PostHog React documentation:
 * https://posthog.com/docs/libraries/react
 */
import posthog from 'posthog-js'

export const initPostHog = () => {
  const apiKey = import.meta.env.VITE_POSTHOG_API_KEY
  const host = import.meta.env.VITE_POSTHOG_HOST

  if (!apiKey || !host) {
    console.warn('[PostHog] Analytics disabled: Missing VITE_POSTHOG_API_KEY or VITE_POSTHOG_HOST')
    return null
  }

  const isEmbed = typeof window !== 'undefined' && window.location.pathname.startsWith('/embed/')

  posthog.init(apiKey, {
    api_host: host,
    ui_host: 'https://us.posthog.com', // PostHog dashboard for deep links

    // Session recording for debugging user issues
    // Disabled in development to prevent high CPU usage from recording workers
    disable_session_recording: import.meta.env.DEV,
    session_recording: {
      maskAllInputs: true, // Mask sensitive form inputs
      maskTextSelector: '[data-private]', // Mask elements with data-private attribute
    },

    // Disable surveys in embed context
    disable_surveys: isEmbed,

    // Autocapture settings — disabled entirely in embed context
    autocapture: isEmbed ? false : {
      dom_event_allowlist: ['click', 'submit', 'change'], // Only capture these events
      element_allowlist: ['button', 'a', 'form', 'input'], // Only from these elements
    },

    // Performance optimization
    loaded: () => {
      if (import.meta.env.DEV) {
        console.log('[PostHog] Analytics initialized')
      }
    },

    // Respect user privacy
    respect_dnt: true, // Respect Do Not Track browser setting

    // Persistence
    persistence: 'localStorage', // Use localStorage for session persistence

    // Debugging (only in development)
    debug: import.meta.env.DEV,
  })

  return posthog
}

export { posthog }
