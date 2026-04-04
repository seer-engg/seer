import { useState, useEffect, useMemo } from 'react';
import { useConnectIntegration } from '@/hooks/useConnectIntegration';
import { useToolingData } from '@/hooks/useToolingData';
import { useToast } from '@/hooks/utility/use-toast';
import { useSupabaseProjects } from './useSupabaseProjects';
import { useSupabaseHandlers } from './useSupabaseHandlers';

export function useSupabaseBinding() {
  const [bindModalOpen, setBindModalOpen] = useState(false);
  const [bindingProjectId, setBindingProjectId] = useState<string | null>(null);

  const { toast } = useToast();
  const connectIntegration = useConnectIntegration();
  const { getConnectionId, isIntegrationConnected, toolsWithStatus } = useToolingData();

  const supabaseToolNames = useMemo(() => {
    return toolsWithStatus
      .filter(tool => tool.integrationType === 'supabase')
      .map(tool => tool.tool.name);
  }, [toolsWithStatus]);

  const supabaseConnected = isIntegrationConnected('supabase');

  const projects = useSupabaseProjects(bindModalOpen && supabaseConnected);

  // Reset state when modal closes
  useEffect(() => {
    if (!bindModalOpen) {
      setBindingProjectId(null);
    }
  }, [bindModalOpen]);

  const handlers = useSupabaseHandlers({
    supabaseToolNames,
    getConnectionId,
    connectIntegration,
    toast,
    setBindingProjectId,
    setBindModalOpen,
  });

  const isBindingProject = bindingProjectId !== null;

  return {
    bindModalOpen,
    setBindModalOpen,
    bindingSearch: projects.bindingSearch,
    setBindingSearch: projects.setBindingSearch,
    bindingDebouncedSearch: projects.bindingDebouncedSearch,
    bindingProjectId,
    isBindingProject,
    supabaseConnected,
    supabaseToolNames,
    supabaseProjectItems: projects.items,
    supabaseProjectsLoading: projects.isLoading,
    supabaseProjectsIsError: projects.isError,
    supabaseProjectsError: projects.error,
    supabaseProjectsHasNextPage: projects.hasNextPage,
    supabaseProjectsFetchingNextPage: projects.isFetchingNextPage,
    fetchNextSupabaseProjects: projects.fetchNextPage,
    handleConnectSupabase: handlers.handleConnectSupabase,
    handleBindProject: handlers.handleBindProject,
  };
}
