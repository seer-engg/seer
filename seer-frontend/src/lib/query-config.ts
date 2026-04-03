import type { QueryClientConfig } from '@tanstack/react-query';
import { BackendAPIError } from './api-client';

/**
 * HTTP status codes that should NOT be retried
 * These represent permanent failures where retrying won't help
 */
const NON_RETRYABLE_STATUS_CODES = [
  400, // Bad Request - client error won't be fixed by retry
  401, // Unauthorized - need to re-authenticate
  402, // Payment Required - need to upgrade plan
  403, // Forbidden - insufficient permissions
  404, // Not Found - resource doesn't exist
  409, // Conflict - handled by custom logic
  422, // Unprocessable Entity - validation error
  500, // Internal Server Error - server-side issue
  502, // Bad Gateway - upstream server issue
  503, // Service Unavailable - maintenance or overload
];

/**
 * HTTP status codes that SHOULD be retried
 * These represent temporary failures that may resolve on retry
 */
const RETRYABLE_STATUS_CODES = [
  408, // Request Timeout - temporary timeout
  429, // Too Many Requests - rate limit (with backoff)
];

/**
 * Determines if a failed request should be retried based on the error and attempt count
 *
 * @param failureCount - Number of failed attempts so far (0-indexed)
 * @param error - The error that occurred
 * @returns true if the request should be retried, false otherwise
 */
export function shouldRetryRequest(failureCount: number, error: unknown): boolean {
  // Never retry more than 3 times
  if (failureCount >= 3) {
    return false;
  }

  // If it's a BackendAPIError with a status code, check our rules
  if (error instanceof BackendAPIError && error.status) {
    // Don't retry if it's a non-retryable status code
    if (NON_RETRYABLE_STATUS_CODES.includes(error.status)) {
      return false;
    }

    // Do retry if it's a retryable status code
    if (RETRYABLE_STATUS_CODES.includes(error.status)) {
      return true;
    }

    // For any other status code, don't retry (fail fast)
    return false;
  }

  // For network errors (no status code), retry with backoff
  // These are usually connection issues that may be temporary
  return true;
}

/**
 * Calculates the retry delay using exponential backoff
 *
 * @param attemptIndex - The attempt number (0-indexed)
 * @returns Delay in milliseconds
 */
export function getRetryDelay(attemptIndex: number): number {
  // Exponential backoff: 1s, 2s, 4s, 8s, etc.
  // Cap at 30 seconds to avoid excessive delays
  const baseDelay = 1000; // 1 second
  const exponentialDelay = baseDelay * Math.pow(2, attemptIndex);
  return Math.min(exponentialDelay, 30000); // Max 30 seconds
}

/**
 * Default React Query configuration with smart retry logic
 * Prevents recursive retries for non-retryable errors (400, 402, 500, etc.)
 */
export const defaultQueryOptions: QueryClientConfig['defaultOptions'] = {
  queries: {
    retry: shouldRetryRequest,
    retryDelay: getRetryDelay,
  },
  mutations: {
    // Mutations typically shouldn't retry automatically
    // Most mutations are not idempotent and retrying could cause duplicate actions
    retry: false,
  },
};
