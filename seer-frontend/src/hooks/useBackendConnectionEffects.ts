import { useEffect, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { toast } from '@/components/ui/sonner';
import { useBackendHealth } from '@/lib/backend-health';
import { getStoredBackendBaseUrl, setStoredBackendBaseUrl } from '@/lib/api-client';
import { useRefreshToolingData } from './useToolingData';

export function useBackendConnectionEffects() {
  const { isHealthy } = useBackendHealth();
  const [searchParams, setSearchParams] = useSearchParams();
  const refreshToolingData = useRefreshToolingData();

  // Track previous health state to detect actual status transitions
  const prevHealthyRef = useRef<boolean | null>(null);

  // Handle backend URL query parameter - persist to localStorage
  useEffect(() => {
    const backendParam = searchParams.get('backend');
    if (backendParam) {
      const currentStored = getStoredBackendBaseUrl();

      // Only persist and show toast if different from current
      if (currentStored !== backendParam) {
        setStoredBackendBaseUrl(backendParam);
        void refreshToolingData();

        toast.success('Backend Connected', {
          description: `Connected to backend at ${backendParam}`,
          duration: 4000,
        });
      }

      // Remove query parameter for clean URL
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete('backend');
      setSearchParams(newSearchParams, { replace: true });
    }
  }, [refreshToolingData, searchParams, setSearchParams]);

  // Show toast notifications for backend status changes
  // Only show toast on actual status TRANSITIONS, not on every poll
  useEffect(() => {
    const prevHealthy = prevHealthyRef.current;

    // Only show toast if status actually changed (not on initial load or repeated polls)
    if (prevHealthy !== null && prevHealthy !== isHealthy) {
      if (isHealthy === true) {
        toast.success('Backend Connected', {
          description: 'Successfully connected to backend server',
          duration: 3000,
        });
      } else if (isHealthy === false) {
        toast.error('Backend Disconnected', {
          description: 'Unable to connect to backend server',
          duration: 5000,
        });
      }
    }

    // Update the ref to track current state for next comparison
    prevHealthyRef.current = isHealthy;
  }, [isHealthy]);

  // Handle OAuth connection redirect - refresh integration status when connected
  useEffect(() => {
    const connectedParam = searchParams.get('connected');
    if (connectedParam) {
      // Refresh integration tools status to reflect new connection
      void refreshToolingData();

      // Show success toast
      toast.success('Integration Connected', {
        description: `Successfully connected to ${connectedParam}`,
        duration: 3000,
      });

      // Remove query parameter from URL
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete('connected');
      setSearchParams(newSearchParams, { replace: true });
    }
  }, [searchParams, refreshToolingData, setSearchParams]);
}
