import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import type { IntegrationResource } from '@/lib/api-client';
import type { useSupabaseBinding } from './hooks/useSupabaseBinding';
import type { ResourceItem } from './types';
import { SupabaseConnectionStep } from './SupabaseBindingDialog/SupabaseConnectionStep';
import { SupabaseProjectSelectionStep } from './SupabaseBindingDialog/SupabaseProjectSelectionStep';

export interface SupabaseBindingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bindingState: ReturnType<typeof useSupabaseBinding>;
  onBindSuccess: (resource: IntegrationResource, fallback?: { displayName?: string; projectRef?: string }) => void;
  refetchResources: () => void;
}

export function SupabaseBindingDialog({
  open,
  onOpenChange,
  bindingState,
  onBindSuccess,
  refetchResources,
}: SupabaseBindingDialogProps) {
  const {
    bindingSearch,
    setBindingSearch,
    supabaseConnected,
    supabaseProjectItems,
    supabaseProjectsLoading,
    supabaseProjectsIsError,
    supabaseProjectsError,
    supabaseProjectsHasNextPage,
    supabaseProjectsFetchingNextPage,
    fetchNextSupabaseProjects,
    handleConnectSupabase,
    handleBindProject,
    bindingProjectId,
  } = bindingState;

  const bindingDebouncedSearch = bindingState.bindingDebouncedSearch;

  const handleProjectBind = (project: ResourceItem) => {
    handleBindProject(
      Number(project.id),
      project.project_ref || project.id,
      project.display_name,
      onBindSuccess,
      refetchResources
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle>Bind Supabase Project</DialogTitle>
          <DialogDescription>
            {supabaseConnected
              ? 'Select a project to use with this trigger.'
              : 'Connect your Supabase account to browse your projects.'}
          </DialogDescription>
        </DialogHeader>

        {!supabaseConnected ? (
          <SupabaseConnectionStep onConnect={handleConnectSupabase} />
        ) : (
          <SupabaseProjectSelectionStep
            items={supabaseProjectItems}
            isLoading={supabaseProjectsLoading}
            isError={supabaseProjectsIsError}
            error={supabaseProjectsError}
            debouncedSearch={bindingDebouncedSearch}
            searchValue={bindingSearch}
            onSearchChange={setBindingSearch}
            selectedProjectId={bindingProjectId}
            onSelectProject={handleProjectBind}
            hasNextPage={supabaseProjectsHasNextPage}
            isFetchingNextPage={supabaseProjectsFetchingNextPage}
            onLoadMore={fetchNextSupabaseProjects}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}
