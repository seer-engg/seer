import { useStartIntegrationConnection } from './useStartIntegrationConnection';

/**
 * Simplified trigger handlers hook - Phase 2 refactoring
 * No more draft system, triggers are managed directly in store with autosave
 */
export function useTriggerHandlers({
  gmailToolNames,
  discordToolNames,
  slackToolNames,
  calendarToolNames,
}: {
  gmailToolNames: string[];
  discordToolNames: string[];
  slackToolNames: string[];
  calendarToolNames: string[];
}) {
  const {
    isConnecting: isConnectingGmail,
    startConnection: handleConnectGmail,
  } = useStartIntegrationConnection({
    integrationType: 'gmail',
    label: 'Gmail',
    toolNames: gmailToolNames,
  });

  const {
    isConnecting: isConnectingDiscord,
    startConnection: handleConnectDiscord,
  } = useStartIntegrationConnection({
    integrationType: 'discord',
    label: 'Discord',
    toolNames: discordToolNames,
  });

  const {
    isConnecting: isConnectingSlack,
    startConnection: handleConnectSlack,
  } = useStartIntegrationConnection({
    integrationType: 'slack',
    label: 'Slack',
    toolNames: slackToolNames,
  });

  const {
    isConnecting: isConnectingCalendar,
    startConnection: handleConnectCalendar,
  } = useStartIntegrationConnection({
    integrationType: 'google_calendar',
    label: 'Google Calendar',
    toolNames: calendarToolNames,
  });

  return {
    isConnectingGmail,
    handleConnectGmail,
    isConnectingDiscord,
    handleConnectDiscord,
    isConnectingSlack,
    handleConnectSlack,
    isConnectingCalendar,
    handleConnectCalendar,
  };
}
