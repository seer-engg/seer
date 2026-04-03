import { CALENDAR_TOOL_FALLBACK_NAMES } from '@/components/workflows/triggers/constants';
import { useProviderIntegrationData } from './useProviderIntegrationData';

/**
 * Hook for Google Calendar integration OAuth flow.
 *
 * Note: Connection status (is_connected) is now provided by the backend
 * via trigger descriptors. This hook only provides data needed for the
 * OAuth connect flow (connectionId, toolNames).
 */
export function useCalendarIntegration() {
  const { toolNames: calendarToolNames, connectionId: calendarConnectionId } = useProviderIntegrationData({
    integrationType: 'google_calendar',
    fallbackToolNames: CALENDAR_TOOL_FALLBACK_NAMES,
  });

  return {
    calendarToolNames,
    calendarConnectionId,
  };
}
