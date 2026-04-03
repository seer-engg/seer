/**
 * Sentry Error Monitoring Configuration
 *
 * Initializes Sentry for error tracking, performance monitoring,
 * and session replay. Based on Sentry React documentation.
 * @see https://docs.sentry.io/platforms/javascript/guides/react/
 */
import * as Sentry from '@sentry/react';

/**
 * Initialize Sentry error monitoring.
 * Should be called before React renders to catch early errors.
 */
export const initSentry = (): typeof Sentry | null => {
  const dsn = import.meta.env.VITE_SENTRY_DSN;

  if (!dsn) {
    if (import.meta.env.DEV) {
      console.warn('[Sentry] Error monitoring disabled: Missing VITE_SENTRY_DSN');
    }
    return null;
  }

  Sentry.init({
    dsn,
    environment: import.meta.env.MODE,
    release: import.meta.env.VITE_APP_VERSION || undefined,

    // Performance Monitoring
    // 10% in production, 100% in development
    tracesSampleRate: import.meta.env.PROD ? 0.1 : 1.0,

    // Session Replay DISABLED - using PostHog session recording instead
    // This reduces CPU overhead by avoiding duplicate recording workers

    integrations: [
      Sentry.browserTracingIntegration(),
      // replayIntegration removed - PostHog handles session recording
    ],

    // Filter out noisy errors
    ignoreErrors: [
      // Browser extensions
      /^chrome-extension:\/\//,
      /^moz-extension:\/\//,
      // Network errors that are usually transient
      'Failed to fetch',
      'NetworkError',
      'Load failed',
      // AbortController cancellations
      'AbortError',
      // User cancelled operations
      'cancelled',
      // ResizeObserver loop errors (benign)
      'ResizeObserver loop limit exceeded',
      'ResizeObserver loop completed with undelivered notifications',
    ],

    beforeSend(event, hint) {
      // Don't send in development unless debug is enabled
      if (import.meta.env.DEV && !import.meta.env.VITE_SENTRY_DEBUG) {
        console.log('[Sentry] Would send event:', {
          message: event.message,
          exception: hint.originalException,
        });
        return null;
      }

      return event;
    },

    // Debugging (only in development with VITE_SENTRY_DEBUG=true)
    debug: import.meta.env.DEV && !!import.meta.env.VITE_SENTRY_DEBUG,
  });

  if (import.meta.env.DEV) {
    console.log('[Sentry] Error monitoring initialized');
  }

  return Sentry;
};

export { Sentry };
