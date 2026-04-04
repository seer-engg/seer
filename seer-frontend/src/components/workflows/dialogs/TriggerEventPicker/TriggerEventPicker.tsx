/**
 * TriggerEventPicker Component
 *
 * A dialog for browsing and selecting real trigger events (emails, messages,
 * webhooks, forms) from the user's connected accounts or stored events
 * for workflow testing.
 */

import { useState, useCallback, useMemo } from 'react';
import { RefreshCw } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { TriggerEventList } from './TriggerEventList';
import { useTriggerEventFetch } from './hooks/useTriggerEventFetch';
import { TRIGGER_DISPLAY_CONFIG, TRIGGER_BROWSING_MODE } from './types';
import type { TriggerEventPickerProps } from './types';
import type { TriggerEventItem } from '@/types/triggers';
import { cn } from '@/lib/utils';

function getMissingMessage(triggerKey: TriggerEventPickerProps['triggerKey']) {
  const mode = TRIGGER_BROWSING_MODE[triggerKey];
  if (mode === 'polling') {
    return 'Missing connection. Please connect your account first.';
  }
  return 'No events received yet. Send a test event first.';
}

interface DialogBodyProps {
  triggerKey: TriggerEventPickerProps['triggerKey'];
  browsingMode: string;
  shouldDisableFetcher: boolean;
  isLoading: boolean;
  refetch: () => void;
  items: TriggerEventItem[];
  selectedEventId: string | undefined;
  onSelect: (event: TriggerEventItem) => void;
  hasNextPage: boolean;
  fetchNextPage: () => void;
  isFetchingNextPage: boolean;
  isError: boolean;
  error: Error | null;
  selectedEvent: TriggerEventItem | undefined;
  onConfirm: () => void;
  onClose: () => void;
}

function TriggerEventPickerBody({
  triggerKey,
  browsingMode,
  shouldDisableFetcher,
  isLoading,
  refetch,
  items,
  selectedEventId,
  onSelect,
  hasNextPage,
  fetchNextPage,
  isFetchingNextPage,
  isError,
  error,
  selectedEvent,
  onConfirm,
  onClose,
}: DialogBodyProps) {
  const config = TRIGGER_DISPLAY_CONFIG[triggerKey];
  const title = config?.label ? `Select ${config.label}` : 'Select Event';
  const description =
    browsingMode === 'persisted'
      ? 'Choose from recently received events to test the workflow.'
      : 'Choose a recent event from your connected account to test the workflow.';

  return (
    <>
      <DialogHeader>
        <div className="flex items-start justify-between gap-2">
          <div>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>{description}</DialogDescription>
          </div>
          {!shouldDisableFetcher && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => refetch()}
              disabled={isLoading}
              aria-label="Refresh events"
              className="h-8 w-8 shrink-0 mt-0.5"
            >
              <RefreshCw className={cn('h-4 w-4', isLoading && 'animate-spin')} />
            </Button>
          )}
        </div>
      </DialogHeader>

      {shouldDisableFetcher ? (
        <div className="p-4 text-center text-muted-foreground text-sm">{getMissingMessage(triggerKey)}</div>
      ) : (
        <>
          <TriggerEventList
            items={items}
            selectedEventId={selectedEventId}
            onSelect={onSelect}
            hasNextPage={hasNextPage}
            fetchNextPage={fetchNextPage}
            isFetchingNextPage={isFetchingNextPage}
            isLoading={isLoading}
            isError={isError}
            error={error}
            refetch={refetch}
            triggerKey={triggerKey}
          />
          <DialogFooter>
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={onConfirm} disabled={!selectedEvent}>
              Use this event
            </Button>
          </DialogFooter>
        </>
      )}
    </>
  );
}

export function TriggerEventPicker({
  open,
  onOpenChange,
  triggerKey,
  provider,
  providerConnectionId,
  subscriptionId,
  filterParams,
  onEventSelect,
}: TriggerEventPickerProps) {
  const [selectedEventId, setSelectedEventId] = useState<string | undefined>();

  const {
    items,
    isLoading,
    isError,
    error,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
    refetch,
    shouldDisableFetcher,
    browsingMode,
  } = useTriggerEventFetch({
    provider,
    triggerKey,
    providerConnectionId,
    subscriptionId,
    filterParams,
    open,
  });

  const handleSelect = useCallback(
    (event: TriggerEventItem) => {
      setSelectedEventId(event.id);
    },
    [],
  );

  const selectedEvent = useMemo(
    () => items.find((item) => item.id === selectedEventId),
    [items, selectedEventId],
  );

  const handleConfirm = useCallback(() => {
    if (selectedEvent) {
      onEventSelect(selectedEvent);
      onOpenChange(false);
    }
  }, [selectedEvent, onEventSelect, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <TriggerEventPickerBody
          triggerKey={triggerKey}
          browsingMode={browsingMode}
          shouldDisableFetcher={shouldDisableFetcher}
          isLoading={isLoading}
          refetch={refetch}
          items={items}
          selectedEventId={selectedEventId}
          onSelect={handleSelect}
          hasNextPage={hasNextPage}
          fetchNextPage={fetchNextPage}
          isFetchingNextPage={isFetchingNextPage}
          isError={isError}
          error={error}
          selectedEvent={selectedEvent}
          onConfirm={handleConfirm}
          onClose={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
