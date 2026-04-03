import { Search, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import type { ResourceItem } from '../types';

export interface SupabaseProjectSelectionStepProps {
  items: ResourceItem[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  debouncedSearch: string;
  searchValue: string;
  onSearchChange: (value: string) => void;
  selectedProjectId: string | null;
  onSelectProject: (project: ResourceItem) => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  onLoadMore: () => void;
}

export function SupabaseProjectSelectionStep({
  items,
  isLoading,
  isError,
  error,
  debouncedSearch,
  searchValue,
  onSearchChange,
  selectedProjectId,
  onSelectProject,
  hasNextPage,
  isFetchingNextPage,
  onLoadMore,
}: SupabaseProjectSelectionStepProps) {
  return (
    <>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search projects..."
          value={searchValue}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-9"
        />
      </div>
      <ScrollArea className="mt-3 max-h-[320px] pr-2">
        {isLoading && items.length === 0 ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-14 w-full" />
            ))}
          </div>
        ) : isError ? (
          <div className="p-4 text-center text-destructive text-sm">
            {error instanceof Error ? error.message : 'Unable to load Supabase projects'}
          </div>
        ) : items.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground text-sm">
            {debouncedSearch ? 'No matching projects.' : 'No projects available.'}
          </div>
        ) : (
          <div className="space-y-2">
            {items.map((project) => (
              <div
                key={project.id}
                className="flex items-center justify-between gap-3 p-3 border rounded-md"
              >
                <div className="min-w-0">
                  <p className="font-medium text-sm truncate">{project.display_name}</p>
                  <p className="text-xs text-muted-foreground truncate">
                    {project.project_ref || project.id}
                    {project.region ? ` • ${project.region}` : ''}
                  </p>
                </div>
                <Button
                  size="sm"
                  onClick={() => onSelectProject(project)}
                  disabled={selectedProjectId !== null && selectedProjectId !== project.id}
                >
                  {selectedProjectId === project.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    'Bind'
                  )}
                </Button>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
      {hasNextPage && (
        <Button
          variant="ghost"
          className="w-full mt-2"
          onClick={onLoadMore}
          disabled={isFetchingNextPage}
        >
          {isFetchingNextPage ? 'Loading...' : 'Load more'}
        </Button>
      )}
    </>
  );
}
