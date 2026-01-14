// Google Analytics 4 event tracking utilities
// Only active in production builds

/**
 * Fire a custom GA4 event
 */
export function trackEvent(eventName: string, eventParams?: Record<string, any>) {
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', eventName, eventParams);
  }
}

/**
 * Track search query events from Algolia
 */
export function trackSearchQuery(query: string, resultCount?: number) {
  trackEvent('search_query', {
    search_term: query,
    result_count: resultCount,
  });
}

/**
 * Track external link clicks
 */
export function trackExternalLinkClick(url: string, linkText?: string) {
  trackEvent('external_link_click', {
    link_url: url,
    link_text: linkText,
  });
}

/**
 * Track code block copy events
 */
export function trackCodeCopy(language?: string, codePreview?: string) {
  trackEvent('code_copy', {
    language: language || 'unknown',
    code_preview: codePreview || '',
  });
}

/**
 * Track scroll depth milestones
 */
export function trackScrollDepth(depth: number) {
  trackEvent('scroll_depth', {
    scroll_percentage: depth,
  });
}

/**
 * Initialize custom event tracking handlers
 */
export function initializeCustomTracking() {
  // Only initialize in production (GA4 plugin handles this, but be explicit)
  if (process.env.NODE_ENV !== 'production') {
    return;
  }

  // Track external links
  trackExternalLinks();

  // Track scroll depth
  trackScrollDepth_();

  // Algolia search tracking is handled separately via docusaurus search integration
}

/**
 * Add click tracking to external links
 */
function trackExternalLinks() {
  document.addEventListener('click', (event) => {
    const target = event.target as HTMLElement;
    const link = target.closest('a');

    if (!link) return;

    const href = link.getAttribute('href');
    const text = link.textContent || '';

    // Only track external links (not same-origin)
    if (href && !href.startsWith('/') && !href.startsWith('#')) {
      trackExternalLinkClick(href, text.trim());
    }
  });
}

/**
 * Track scroll depth at 25%, 50%, 75%, and 100%
 */
function trackScrollDepth_() {
  const trackingThresholds = [25, 50, 75, 100];
  const tracked = new Set<number>();

  window.addEventListener('scroll', () => {
    if (document.body.scrollHeight === 0) return;

    const scrollPercentage = Math.round(
      (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100
    );

    for (const threshold of trackingThresholds) {
      if (scrollPercentage >= threshold && !tracked.has(threshold)) {
        tracked.add(threshold);
        trackScrollDepth(threshold);
      }
    }
  });
}

// Extend Window interface for TypeScript
declare global {
  interface Window {
    gtag?: (...args: any[]) => void;
  }
}
