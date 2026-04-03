import { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { backendApiClient, getUserSettings } from '@/lib/api-client';
import { modelKeys, userKeys } from '@/lib/query-keys';
import { useChatStore } from '@/stores';
import type { ModelInfo } from '@/components/workflows/buildtypes';
import { FALLBACK_MODEL, setUserDefaultModel } from '@/lib/default-model';

export function useAvailableModels() {
  const selectedModel = useChatStore((state) => state.selectedModel);
  const setSelectedModel = useChatStore((state) => state.setSelectedModel);

  const { data: models = [], isLoading: isLoadingModels } = useQuery<ModelInfo[]>({
    queryKey: modelKeys.available(),
    queryFn: async () => {
      const response = await backendApiClient.request<ModelInfo[]>(
        '/api/models',
        {
          method: 'GET',
        }
      );
      return response;
    },
  });

  const { data: settings } = useQuery({
    queryKey: userKeys.settings(),
    queryFn: getUserSettings,
  });

  // Keep module-level default in sync for non-React code (workflow utils)
  useEffect(() => {
    const userDefault = settings?.preferences?.default_model as string | undefined;
    if (userDefault) {
      setUserDefaultModel(userDefault);
    }
  }, [settings]);

  useEffect(() => {
    if (models.length > 0 && !selectedModel) {
      const userDefault = settings?.preferences?.default_model as string | undefined;
      const preferredId = userDefault || FALLBACK_MODEL;
      const preferredModel =
        models.find((m) => m.id === preferredId) || models.find((m) => m.id === FALLBACK_MODEL) || models[0];
      setSelectedModel(preferredModel.id);
    }
  }, [models, settings, selectedModel, setSelectedModel]);

  return { models, isLoadingModels };
}
