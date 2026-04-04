/**
 * List of trigger events with loading, error, and empty states.
 * Follows the ResourceList pattern from ResourcePicker.
 */

import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { TriggerEventListItem } from './TriggerEventListItem';
import type { TriggerEventListProps } from './types';

interface TriggerEventListPropsExtended extends TriggerEventListProps {
  triggerKey: string;
}

export function TriggerEventList({
  items,
  selectedEventId,
  onSelect,
  hasNextPage,
  fetchNextPage,
  isFetchingNextPage,
  isLoading,
  isError,
  error,
  refetch,
  triggerKey,
}: TriggerEventListPropsExtended) {
  return (
    <ScrollArea className="flex-1 min-h-[300px] max-h-[400px]">
      {isLoading && items.length === 0 ? (
        <div className="space-y-2 p-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="space-y-2 p-3">
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-3 w-1/2" />
              <Skeleton className="h-3 w-full" />
            </div>
          ))}
        </div>
      ) : isError ? (
        <div className="p-4 text-center text-destructive">
          <p>Error loading events</p>
          <p className="text-sm text-muted-foreground mt-1">
            {error instanceof Error ? error.message : 'Unknown error'}
          </p>
          <Button variant="outline" size="sm" className="mt-2" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      ) : items.length === 0 ? (
        <div className="p-4 text-center text-muted-foreground">
          <p>No events found</p>
          <p className="text-sm mt-1">No recent events in your connected account</p>
        </div>
      ) : (
        <div className="space-y-1 p-2">
          {items.map((item) => (
            <TriggerEventListItem
              key={item.id}
              item={item}
              isSelected={selectedEventId === item.id}
              onSelect={onSelect}
              triggerKey={triggerKey}
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
