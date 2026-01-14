// Client-side module for initializing custom analytics tracking
// Automatically loaded by Docusaurus

import { initializeCustomTracking, trackSearchQuery } from '../utils/analytics';

export function onRouteDidUpdate() {
  // Initialize custom tracking on every route change
  initializeCustomTracking();

  // Track Algolia search events
  trackAlgoliaSearch();
}

/**
 * Set up tracking for Algolia search queries
 */
function trackAlgoliaSearch() {
  // Wait for Algolia/DocSearch to load
  const checkAlgolia = setInterval(() => {
    if (typeof window !== 'undefined' && (window as any).docsearch) {
      clearInterval(checkAlgolia);

      // Hook into DocSearch events if available
      const searchInputs = document.querySelectorAll('input[placeholder*="Search"]');
      searchInputs.forEach((input) => {
        input.addEventListener('keydown', (e) => {
          if ((e as KeyboardEvent).key === 'Enter') {
            const query = (e.target as HTMLInputElement).value;
            if (query.trim()) {
              trackSearchQuery(query);
            }
          }
        });
      });
    }
  }, 500);

  // Stop checking after 5 seconds if Algolia doesn't load
  setTimeout(() => clearInterval(checkAlgolia), 5000);
}
