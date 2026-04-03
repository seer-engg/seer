/**
 * Sentry Error Boundary Component
 *
 * Catches React rendering errors and reports them to Sentry.
 * Shows a user-friendly fallback UI when errors occur.
 */
import * as Sentry from '@sentry/react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FallbackProps {
  error: Error;
  componentStack: string | null;
  eventId: string | null;
  resetError: () => void;
}

const ErrorFallback = ({ error, resetError }: FallbackProps) => {
  const handleGoHome = () => {
    window.location.href = '/';
  };

  const handleReportFeedback = () => {
    Sentry.showReportDialog();
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <div className="max-w-md w-full space-y-6 text-center">
        <div className="flex justify-center">
          <div className="rounded-full bg-destructive/10 p-4">
            <AlertTriangle className="h-8 w-8 text-destructive" />
          </div>
        </div>

        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-foreground">
            Something went wrong
          </h1>
          <p className="text-muted-foreground">
            We've been notified and are looking into it. Please try again or
            return to the dashboard.
          </p>
        </div>

        {import.meta.env.DEV && (
          <div className="rounded-lg border bg-muted/50 p-4 text-left">
            <p className="text-sm font-mono text-destructive break-all">
              {error.message}
            </p>
          </div>
        )}

        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <Button variant="outline" onClick={resetError}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Try Again
          </Button>
          <Button onClick={handleGoHome}>
            <Home className="mr-2 h-4 w-4" />
            Go to Dashboard
          </Button>
        </div>

        <Button
          variant="link"
          className="text-sm text-muted-foreground"
          onClick={handleReportFeedback}
        >
          Report this issue
        </Button>
      </div>
    </div>
  );
};

interface SentryErrorBoundaryProps {
  children: React.ReactNode;
}

export const SentryErrorBoundary = ({ children }: SentryErrorBoundaryProps) => {
  return (
    <Sentry.ErrorBoundary
      fallback={ErrorFallback}
      showDialog
      dialogOptions={{
        title: "It looks like we're having issues.",
        subtitle: 'Our team has been notified.',
        subtitle2:
          "If you'd like to help, tell us what happened below.",
      }}
    >
      {children}
    </Sentry.ErrorBoundary>
  );
};
