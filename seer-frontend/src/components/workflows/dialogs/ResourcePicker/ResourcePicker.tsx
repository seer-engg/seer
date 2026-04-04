/**
 * Resource Picker Component
 *
 * A reusable component for browsing and selecting integration resources
 * like Google Sheets, Drive files, GitHub repos, etc.
 */
import { ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DialogTrigger } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';
import { useSupabaseBinding } from './hooks/useSupabaseBinding';
import { usePostgresBinding } from './hooks/usePostgresBinding';
import { useResourcePicker } from './hooks/useResourcePicker';
import { ResourceDialog } from './ResourceDialog';
import { ResourcePickerContent } from './ResourcePickerContent';
import { SupabaseBindingDialog } from './SupabaseBindingDialog';
import { PostgresBindingDialog } from './PostgresBindingDialog';
import type { ResourcePickerProps } from './types';

export function ResourcePicker({
  config,
  provider,
  value,
  onChange,
  placeholder = 'Select a resource...',
  disabled = false,
  dependsOnValues,
  className,
}: ResourcePickerProps) {
  const picker = useResourcePicker({ config, provider, value, onChange, dependsOnValues });
  const supabaseBinding = useSupabaseBinding();
  const postgresBinding = usePostgresBinding();

  return (
    <>
      <ResourceDialog
        open={picker.open}
        onOpenChange={picker.setOpen}
        trigger={
          <DialogTrigger asChild>
            <Button
              variant="outline"
              className={cn(
                'w-full justify-between text-left font-normal',
                !picker.displayValue && 'text-muted-foreground',
                className,
              )}
              disabled={disabled || picker.hasMissingDependency}
            >
              <span className="truncate">
                {picker.hasMissingDependency
                  ? `Select ${config.depends_on} first`
                  : picker.displayValue || placeholder}
              </span>
              <ChevronRight className="ml-2 h-4 w-4 shrink-0 opacity-50" />
            </Button>
          </DialogTrigger>
        }
      >
        <ResourcePickerContent
          isSupabaseBindingPicker={picker.isSupabaseBindingPicker}
          isPostgresBindingPicker={picker.isPostgresBindingPicker}
          onRefresh={picker.resourceFetch.refetch}
          isRefreshing={picker.resourceFetch.isLoading}
          onOpenBindingDialog={() => supabaseBinding.setBindModalOpen(true)}
          onOpenPostgresBindingDialog={() => postgresBinding.setBindModalOpen(true)}
          currentPath={picker.currentPath}
          onBack={() => {
            picker.setCurrentPath(prev => prev.slice(0, -1));
            picker.setSearchQuery('');
          }}
          onNavigateToPath={(index) => picker.setCurrentPath(prev => prev.slice(0, index + 1))}
          supportsHierarchy={picker.resourceFetch.supportsHierarchy}
          searchQuery={picker.searchQuery}
          onSearchChange={picker.setSearchQuery}
          supportsSearch={picker.resourceFetch.supportsSearch}
          items={picker.resourceFetch.items}
          selectedValue={value}
          onSelect={picker.handleSelect}
          hasNextPage={picker.resourceFetch.hasNextPage}
          fetchNextPage={picker.resourceFetch.fetchNextPage}
          isFetchingNextPage={picker.resourceFetch.isFetchingNextPage}
          isLoading={picker.resourceFetch.isLoading}
          isError={picker.resourceFetch.isError}
          error={picker.resourceFetch.error}
          shouldDisableFetcher={picker.resourceFetch.shouldDisableFetcher}
          refetch={picker.resourceFetch.refetch}
        />
      </ResourceDialog>

      {picker.isSupabaseBindingPicker && (
        <SupabaseBindingDialog
          open={supabaseBinding.bindModalOpen}
          onOpenChange={supabaseBinding.setBindModalOpen}
          bindingState={supabaseBinding}
          onBindSuccess={picker.applySupabaseBindingSelection}
          refetchResources={picker.resourceFetch.refetch}
        />
      )}

      {picker.isPostgresBindingPicker && (
        <PostgresBindingDialog
          open={postgresBinding.bindModalOpen}
          onOpenChange={postgresBinding.setBindModalOpen}
          bindingState={postgresBinding}
          onBindSuccess={(resource) => {
            picker.applySupabaseBindingSelection(resource, { displayName: resource.name || undefined });
          }}
          refetchResources={picker.resourceFetch.refetch}
        />
      )}
    </>
  );
}
