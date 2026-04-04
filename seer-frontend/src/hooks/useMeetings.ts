import { useQuery } from '@tanstack/react-query';
import { useAuthStatus } from '@/hooks/useAuthProvider';
import { meetingsApi } from '@/lib/meetings-api';
import { meetingKeys } from '@/lib/query-keys';

function useQueryEnabled() {
  const { isLoaded, isSignedIn } = useAuthStatus();
  return isLoaded && isSignedIn;
}

export function useMeetings() {
  const enabled = useQueryEnabled();
  const { data, isLoading } = useQuery({
    queryKey: meetingKeys.list(),
    queryFn: () => meetingsApi.list(),
    enabled,
  });

  return {
    meetings: data?.items ?? [],
    isLoading,
  };
}

export function useMeetingTranscript(botId: string | null) {
  const enabled = useQueryEnabled();
  const { data, isLoading } = useQuery({
    queryKey: meetingKeys.transcript(botId),
    queryFn: () => meetingsApi.getTranscript(botId!),
    enabled: enabled && !!botId,
  });

  return {
    utterances: data?.utterances ?? [],
    isLoading,
  };
}

export function useMeetingRecording(botId: string | null) {
  const enabled = useQueryEnabled();
  const { data, isLoading } = useQuery({
    queryKey: meetingKeys.recording(botId),
    queryFn: () => meetingsApi.getRecording(botId!),
    enabled: enabled && !!botId,
    staleTime: 0, // Recording URLs are short-lived presigned URLs
  });

  return {
    recording: data ?? null,
    isLoading,
  };
}
