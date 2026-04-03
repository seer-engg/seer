/**
 * Sentry Route Tracker Component
 *
 * Tracks route changes for Sentry performance monitoring.
 * Adds breadcrumbs for navigation events.
 */
import { useEffect } from 'react';
import { useLocation, useNavigationType } from 'react-router-dom';
import * as Sentry from '@sentry/react';

export const SentryRouteTracker = () => {
  const location = useLocation();
  const navigationType = useNavigationType();

  useEffect(() => {
    // Add breadcrumb for navigation
    Sentry.addBreadcrumb({
      category: 'navigation',
      message: `Navigated to ${location.pathname}`,
      level: 'info',
      data: {
        from: document.referrer,
        to: location.pathname + location.search,
        type: navigationType,
      },
    });

    // Set route context for error reports
    Sentry.setContext('route', {
      pathname: location.pathname,
      search: location.search,
      navigationType,
    });
  }, [location, navigationType]);

  return null;
};
