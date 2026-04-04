/**
 * UTM Parameter Tracker
 * Captures UTM parameters from URL and stores them for signup attribution
 */
import { posthog } from '@/lib/posthog'

const STORAGE_KEY = 'seer_signup_source';
const STORAGE_TIMESTAMP_KEY = 'seer_signup_source_timestamp';
const EXPIRY_DAYS = 30; // UTM params expire after 30 days

type SignupSource = 'HN' | 'Twitter' | 'Reddit' | 'LinkedIn' | 'YouTube' | 'Organic' | 'Direct';

/**
 * Map utm_source values to our standardized signup_source values
 */
const UTM_SOURCE_MAP: Record<string, SignupSource> = {
  hackernews: 'HN',
  hn: 'HN',
  twitter: 'Twitter',
  reddit: 'Reddit',
  linkedin: 'LinkedIn',
  youtube: 'YouTube',
  organic: 'Organic',
  google: 'Organic',
  bing: 'Organic',
};

/**
 * Capture UTM parameters from URL and store in localStorage
 * Should be called on app initialization
 */
export function captureUTMParams(): void {
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const utmSource = urlParams.get('utm_source')?.toLowerCase();
    const utmMedium = urlParams.get('utm_medium');
    const utmCampaign = urlParams.get('utm_campaign');

    if (utmSource) {
      // Map UTM source to our standardized values
      const signupSource = UTM_SOURCE_MAP[utmSource] || 'Direct';

      // Store with timestamp
      localStorage.setItem(STORAGE_KEY, signupSource);
      localStorage.setItem(STORAGE_TIMESTAMP_KEY, Date.now().toString());

      // Send to PostHog for unified tracking
      if (posthog.__loaded) {
        posthog.capture('utm_params_captured', {
          utm_source: utmSource,
          utm_medium: utmMedium,
          utm_campaign: utmCampaign,
          signup_source: signupSource,
        })
      }

      console.log(`[UTM Tracker] Captured signup source: ${signupSource} from utm_source=${utmSource}`);
    } else if (!localStorage.getItem(STORAGE_KEY)) {
      // No UTM param and nothing stored = Direct traffic
      localStorage.setItem(STORAGE_KEY, 'Direct');
      localStorage.setItem(STORAGE_TIMESTAMP_KEY, Date.now().toString());

      console.log('[UTM Tracker] No UTM params, defaulting to Direct');
    }
  } catch (error) {
    console.error('[UTM Tracker] Error capturing UTM params:', error);
  }
}

/**
 * Get the stored signup source
 * Returns null if expired or not set
 */
export function getStoredSignupSource(): SignupSource | null {
  try {
    const signupSource = localStorage.getItem(STORAGE_KEY) as SignupSource | null;
    const timestamp = localStorage.getItem(STORAGE_TIMESTAMP_KEY);

    if (!signupSource || !timestamp) {
      return null;
    }

    // Check if expired (30 days)
    const age = Date.now() - parseInt(timestamp, 10);
    const maxAge = EXPIRY_DAYS * 24 * 60 * 60 * 1000;

    if (age > maxAge) {
      clearSignupSource();
      return null;
    }

    return signupSource;
  } catch (error) {
    console.error('[UTM Tracker] Error getting stored signup source:', error);
    return null;
  }
}

/**
 * Clear the stored signup source
 * Should be called after successful signup
 */
export function clearSignupSource(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(STORAGE_TIMESTAMP_KEY);
    console.log('[UTM Tracker] Cleared signup source');
  } catch (error) {
    console.error('[UTM Tracker] Error clearing signup source:', error);
  }
}

/**
 * Check if signup source should be captured from URL
 * Used by ProtectedRoute to update user record
 */
export function shouldCaptureSignupSource(): boolean {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.has('signup_source');
}

/**
 * Get signup source from URL query parameter
 */
export function getSignupSourceFromURL(): SignupSource | null {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('signup_source') as SignupSource | null;
}
