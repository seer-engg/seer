import { toast } from 'sonner';
import * as Sentry from '@sentry/react';
import { BackendAPIError } from './api-client';
import { useUIStore } from '@/stores/uiStore';

/**
 * Error classification types based on HTTP status codes
 */
export type ErrorCategory =
  | 'validation'    // 400, 422 - client validation errors
  | 'auth'          // 401, 403 - authentication/authorization errors
  | 'payment'       // 402 - payment required
  | 'not_found'     // 404 - resource not found
  | 'conflict'      // 409 - conflict (handled by custom logic)
  | 'rate_limit'    // 429 - too many requests
  | 'server'        // 500, 502, 503 - server errors
  | 'timeout'       // 408 - request timeout
  | 'network'       // network connection errors
  | 'unknown';      // unknown errors

/**
 * Context information for error handling
 */
export interface ErrorContext {
  action?: string;              // e.g., 'save workflow', 'execute tool'
  componentName?: string;        // Component where error occurred
  workflowId?: string;          // Related workflow ID
  additionalInfo?: Record<string, unknown>;
}

/**
 * Result of error handling indicating what action to take
 */
export interface ErrorHandlingResult {
  category: ErrorCategory;
  userMessage: string;          // User-friendly message
  technicalMessage?: string;    // Technical details for debugging
  shouldShowToast: boolean;     // Show toast notification
  shouldShowModal: boolean;     // Show modal dialog
  shouldRedirect: boolean;      // Redirect to another page
  redirectPath?: string;        // Redirect destination
  shouldStopPolling: boolean;   // Stop polling queries
  severity: 'error' | 'warning' | 'info';
}

/**
 * Classifies an error into a category based on status code or error type
 */
export function classifyError(error: unknown): ErrorCategory {
  const status = error instanceof BackendAPIError ? error.status : undefined;
  if (typeof status === 'number') {
    return getCategoryFromStatus(status);
  }

  // Check if it's a network error (no status code)
  if (isNetworkError(error)) {
    return 'network';
  }

  return 'unknown';
}

function getCategoryFromStatus(status: number): ErrorCategory {
  if (status >= 500 && status < 600) return 'server';

  const statusMap: Record<number, ErrorCategory> = {
    400: 'validation',
    401: 'auth',
    402: 'payment',
    403: 'auth',
    404: 'not_found',
    408: 'timeout',
    409: 'conflict',
    422: 'validation',
    429: 'rate_limit',
  };

  return statusMap[status] ?? 'unknown';
}

function isNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;

  const message = error.message.toLowerCase();
  return (
    message.includes('network') ||
    message.includes('fetch') ||
    message.includes('connection')
  );
}

/**
 * Generates a user-friendly error message based on error category and context
 */
export function getUserFriendlyMessage(
  category: ErrorCategory,
  error: unknown,
  context?: ErrorContext,
): string {
  const action = context?.action || 'complete the request';

  if (category === 'validation') {
    if (error instanceof BackendAPIError && error.response) {
      const detail = (error.response as { detail?: string }).detail;
      return detail || `Invalid input. Please check your data and try again.`;
    }
    return `Invalid input. Please check your data and try again.`;
  }

  if (category === 'auth') {
    if (error instanceof BackendAPIError && error.status === 401) {
      return 'Your session has expired. Please sign in again.';
    }
    return `You don't have permission to ${action}.`;
  }

  const messages: Record<ErrorCategory, string> = {
    payment: 'Upgrade required to continue',
    not_found: 'The requested resource was not found.',
    conflict: 'A conflict occurred. This is usually handled by custom logic.',
    rate_limit: 'Too many requests. Please wait a moment and try again.',
    server: 'Server error. Please try again later.',
    timeout: 'Request timed out. Please check your connection and try again.',
    network: 'Connection lost. Please check your internet connection.',
    unknown: `Failed to ${action}. Please try again.`,
    validation: '', // handled above
    auth: '', // handled above
  };

  return messages[category] || `Failed to ${action}. Please try again.`;
}

/**
 * Determines if polling should stop based on error type
 * Polling should stop for non-retryable errors
 */
export function shouldStopPolling(error: unknown): boolean {
  const category = classifyError(error);

  // Stop polling for these error types
  const stopPollingCategories: ErrorCategory[] = [
    'validation',   // 400, 422 - won't be fixed by retrying
    'auth',         // 401, 403 - need user action to re-authenticate
    'payment',      // 402 - need payment to continue
    'not_found',    // 404 - resource doesn't exist
    'conflict',     // 409 - conflict state
    'server',       // 500+ - server issues unlikely to resolve quickly
  ];

  return stopPollingCategories.includes(category);
}

/**
 * Checks if the error is a payment required error (402)
 */
export function isPaymentRequiredError(error: unknown): boolean {
  return classifyError(error) === 'payment';
}

/**
 * Main error handling function that determines the appropriate response
 * for a given error
 */
export function handleError(
  error: unknown,
  context?: ErrorContext,
): ErrorHandlingResult {
  const category = classifyError(error);
  const userMessage = getUserFriendlyMessage(category, error, context);

  // Extract technical message for debugging
  let technicalMessage: string | undefined;
  if (error instanceof BackendAPIError) {
    technicalMessage = error.message;
  } else if (error instanceof Error) {
    technicalMessage = error.message;
  }

  // Default result
  const result: ErrorHandlingResult = {
    category,
    userMessage,
    technicalMessage,
    shouldShowToast: true,
    shouldShowModal: false,
    shouldRedirect: false,
    shouldStopPolling: shouldStopPolling(error),
    severity: 'error',
  };

  // Customize response based on category
  switch (category) {
    case 'payment':
      // 402 - Show payment modal instead of toast
      result.shouldShowToast = false;
      result.shouldShowModal = true;
      break;

    case 'auth':
      // 401 - Redirect to sign-in
      if (error instanceof BackendAPIError && error.status === 401) {
        result.shouldRedirect = true;
        result.redirectPath = '/sign-in';
        result.userMessage = 'Session expired. Redirecting to sign in...';
      }
      break;

    case 'conflict':
      // 409 - Don't show toast, let custom handler deal with it
      result.shouldShowToast = false;
      break;

    case 'rate_limit':
      result.severity = 'warning';
      break;

    case 'timeout':
    case 'network':
      result.severity = 'warning';
      break;
  }

  return result;
}

/**
 * Map ErrorCategory to Sentry severity level
 */
function getSentrySeverity(category: ErrorCategory): Sentry.SeverityLevel {
  switch (category) {
    case 'server':
      return 'error';
    case 'auth':
    case 'payment':
    case 'rate_limit':
    case 'timeout':
    case 'network':
      return 'warning';
    case 'validation':
    case 'not_found':
    case 'conflict':
      return 'info';
    default:
      return 'error';
  }
}

/**
 * Report error to Sentry with context
 * Skips validation errors (user errors) and auth errors (expected flow)
 */
function reportToSentry(
  error: unknown,
  category: ErrorCategory,
  context?: ErrorContext,
): void {
  // Don't report validation errors (user errors)
  if (category === 'validation') {
    return;
  }

  // Don't report auth errors (expected flow)
  if (category === 'auth') {
    Sentry.addBreadcrumb({
      category: 'auth',
      message: 'Authentication required',
      level: 'info',
    });
    return;
  }

  const severity = getSentrySeverity(category);

  Sentry.withScope((scope) => {
    // Set error category as tag
    scope.setTag('error_category', category);
    scope.setLevel(severity);

    // Add context
    if (context) {
      scope.setContext('error_context', {
        action: context.action,
        componentName: context.componentName,
        workflowId: context.workflowId,
        ...context.additionalInfo,
      });
    }

    // Add HTTP context for API errors
    if (error instanceof BackendAPIError) {
      scope.setContext('api_error', {
        status: error.status,
        response: error.response,
      });
      scope.setTag('http_status', String(error.status));
    }

    // Capture the error
    if (error instanceof Error) {
      Sentry.captureException(error);
    } else {
      Sentry.captureMessage(String(error), severity);
    }
  });
}

/**
 * Main function to handle and display errors
 * Call this from catch blocks to handle errors consistently
 *
 * @param error - The error that occurred
 * @param context - Optional context about where/why the error occurred
 */
export function showError(error: unknown, context?: ErrorContext): void {
  const result = handleError(error, context);

  // Log technical details for debugging
  if (context?.componentName) {
    console.error(`[${context.componentName}] Error during ${context.action || 'operation'}:`, error);
  } else {
    console.error('Error:', error);
  }

  // Report to Sentry
  reportToSentry(error, result.category, context);

  // Show toast notification
  if (result.shouldShowToast) {
    toast[result.severity === 'error' ? 'error' : result.severity === 'warning' ? 'warning' : 'info'](
      result.userMessage,
      result.technicalMessage ? { description: result.technicalMessage } : undefined,
    );
  }

  // Open payment modal for 402 errors
  if (result.shouldShowModal && result.category === 'payment') {
    useUIStore.getState().setPaymentModalOpen(true);
  }

  // Handle redirects
  if (result.shouldRedirect && result.redirectPath) {
    // Use a small delay to let any toasts show before redirecting
    setTimeout(() => {
      window.location.href = result.redirectPath!;
    }, 500);
  }
}
