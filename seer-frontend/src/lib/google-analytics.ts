/**
 * Google Analytics 4 (GA4) Configuration
 *
 * Backup analytics system alongside PostHog
 * Uses gtag.js for GA4 tracking
 */

// Extend Window interface for gtag
declare global {
  interface Window {
    gtag?: (
      command: 'config' | 'event' | 'set',
      targetId: string,
      config?: Record<string, unknown>
    ) => void
    dataLayer?: unknown[]
  }
}

export const initGoogleAnalytics = (): string | null => {
  const measurementId = import.meta.env.VITE_GA_MEASUREMENT_ID

  if (!measurementId) {
    console.warn('[Google Analytics] Disabled: Missing VITE_GA_MEASUREMENT_ID')
    return null
  }

  // Initialize dataLayer
  window.dataLayer = window.dataLayer || []

  // Define gtag function
  window.gtag = function gtag() {
    // eslint-disable-next-line prefer-rest-params
    window.dataLayer?.push(arguments)
  }

  // Set up initial config
  window.gtag('js', new Date() as unknown as string)
  window.gtag('config', measurementId, {
    send_page_view: true,
  })

  // Load gtag.js script
  const script = document.createElement('script')
  script.async = true
  script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`
  document.head.appendChild(script)

  if (import.meta.env.DEV) {
    console.log(`[Google Analytics] Initialized: ${measurementId}`)
  }

  return measurementId
}

/**
 * Send a custom event to Google Analytics
 */
export const gtag = (
  command: 'config' | 'event' | 'set',
  targetId: string,
  config?: Record<string, unknown>
): void => {
  if (window.gtag) {
    window.gtag(command, targetId, config)
  }
}

/**
 * Track a custom event
 */
export const trackEvent = (
  eventName: string,
  parameters?: Record<string, unknown>
): void => {
  if (window.gtag) {
    window.gtag('event', eventName, parameters)
  }
}

/**
 * Set user properties
 */
export const setUserProperties = (properties: Record<string, unknown>): void => {
  if (window.gtag) {
    window.gtag('set', 'user_properties', properties)
  }
}

/**
 * Set user ID for tracking
 */
export const setUserId = (userId: string | null): void => {
  if (window.gtag) {
    window.gtag('set', 'user_id', { user_id: userId })
  }
}
