import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { ResourceListItem } from './ResourceListItem';
import type { ResourceItem } from './types';

export interface ResourceListProps {
  items: ResourceItem[];
  selectedValue?: string;
  onSelect: (item: ResourceItem) => void;
  supportsHierarchy: boolean;
  hasNextPage: boolean;
  fetchNextPage: () => void;
  isFetchingNextPage: boolean;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  shouldDisableFetcher: boolean;
  debouncedQuery: string;
  refetch: () => void;
}

export function ResourceList({
  items,
  selectedValue,
  onSelect,
  supportsHierarchy,
  hasNextPage,
  fetchNextPage,
  isFetchingNextPage,
  isLoading,
  isError,
  error,
  shouldDisableFetcher,
  debouncedQuery,
  refetch,
}: ResourceListProps) {
  return (
    <ScrollArea className="flex-1 mt-2 min-h-[300px] max-h-[400px]">
      {shouldDisableFetcher ? (
        <div className="p-4 text-center text-destructive text-sm">
          Resource picker misconfigured. Missing provider or endpoint.
        </div>
      ) : isLoading && items.length === 0 ? (
        <div className="space-y-2 p-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-10 w-full" />
          ))}
        </div>
      ) : isError ? (
        <div className="p-4 text-center text-destructive">
          <p>Error loading resources</p>
          <p className="text-sm text-muted-foreground mt-1">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <Button
            variant="outline"
            size="sm"
            className="mt-2"
            onClick={() => refetch()}
          >
            Retry
          </Button>
        </div>
      ) : items.length === 0 ? (
        <div className="p-4 text-center text-muted-foreground">
          {debouncedQuery
            ? 'No results found'
            : 'No resources available'}
        </div>
      ) : (
        <div className="space-y-1 p-2">
          {items.map((item) => (
            <ResourceListItem
              key={item.id}
              item={item}
              isSelected={selectedValue === item.id}
              onSelect={onSelect}
              supportsHierarchy={supportsHierarchy}
            />
          ))}

          {hasNextPage && (
            <Button
              variant="ghost"
              className="w-full mt-2"
              onClick={() => fetchNextPage()}
              disabled={isFetchingNextPage}
            >
              {isFetchingNextPage ? 'Loading...' : 'Load more'}
            </Button>
          )}
        </div>
      )}
    </ScrollArea>
  );
}
