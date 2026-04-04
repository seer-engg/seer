import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { useTriggerCatalogQuery } from '@/hooks/useTriggerCatalogQuery';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';

import { BUILT_IN_BLOCKS } from '../constants';
import { UnifiedBuildItem } from '../build/UnifiedBuildItem';
import { useUnifiedItems } from '@/hooks/useUnifiedItems';
import { useItemSearch } from '@/hooks/useItemSearch';
import { useItemSelection } from '@/hooks/useItemSelection';
import { groupConsecutiveItemsByType } from '@/lib/item-grouping';
import { getLayoutClass, getTypeLabel } from '../build/utils';
import type { TriggerListOption } from '../build/TriggerSection';

/**
 * Centered modal block picker — opens when user clicks "+" on a canvas node or the empty canvas placeholder.
 * Renders as a portal over a dimmed backdrop, centered in the viewport.
 */
// eslint-disable-next-line max-lines-per-function
export function InlineBlockPicker() {
  const inlineBlockPicker = useUIStore((state) => state.inlineBlockPicker);
  const clearInlineBlockPicker = useUIStore((state) => state.clearInlineBlockPicker);
  const clearPendingConnection = useUIStore((state) => state.clearPendingConnection);

  const { data: triggerCatalog = [] } = useTriggerCatalogQuery();

  const [searchQuery, setSearchQuery] = useState('');
  const pickerRef = useRef<HTMLDivElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Convert trigger catalog to simple TriggerListOptions (no action handlers needed for inline picker)
  const triggerOptions = useMemo<TriggerListOption[]>(
    () =>
      triggerCatalog.map((t) => ({
        key: t.key,
        title: t.title,
        description: t.description ?? undefined,
      })),
    [triggerCatalog],
  );

  // Always load all items including triggers - tabs will separate them
  const unifiedItems = useUnifiedItems(BUILT_IN_BLOCKS, triggerOptions, []);
  const filteredItems = useItemSearch(unifiedItems, searchQuery);

  // Separate items by type for tabs
  const blockAndToolItems = useMemo(
    () => filteredItems.filter((item) => item.type !== 'trigger'),
    [filteredItems]
  );
  const triggerItems = useMemo(
    () => filteredItems.filter((item) => item.type === 'trigger'),
    [filteredItems]
  );

  // Group items for rendering within each tab
  const blockAndToolGroups = useMemo(
    () => groupConsecutiveItemsByType(blockAndToolItems),
    [blockAndToolItems]
  );
  const triggerGroups = useMemo(
    () => groupConsecutiveItemsByType(triggerItems),
    [triggerItems]
  );

  // Determine if we're in "add from node" mode (sourceNodeId exists)
  const isAddingFromNode = !!inlineBlockPicker?.sourceNodeId;

  // Default tab: 'blocks' when adding from node, 'blocks' otherwise (triggers accessible via tab)
  const [activeTab, setActiveTab] = useState<'blocks' | 'triggers'>('blocks');

  // On item select: useItemSelection reads pendingConnection from store to place node correctly
  const handleItemClick = useItemSelection();

  const dismiss = useCallback(() => {
    clearInlineBlockPicker();
    clearPendingConnection();
    setSearchQuery('');
  }, [clearInlineBlockPicker, clearPendingConnection]);

  // Focus search and reset tab when picker opens
  useEffect(() => {
    if (inlineBlockPicker?.visible) {
      setSearchQuery('');
      setActiveTab('blocks');
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
  }, [inlineBlockPicker?.visible, inlineBlockPicker?.sourceNodeId]);

  // Dismiss on Escape key
  useEffect(() => {
    if (!inlineBlockPicker?.visible) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        dismiss();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [inlineBlockPicker?.visible, dismiss]);

  if (!inlineBlockPicker?.visible) return null;

  return createPortal(
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-[1px] z-[9998] animate-in fade-in-0 duration-150"
        onClick={dismiss}
        aria-hidden="true"
      />

      {/* Centered modal */}
      <div
        ref={pickerRef}
        role="dialog"
        aria-modal="true"
        aria-label="Select a block to add"
        className={cn(
          'fixed z-[9999] flex flex-col rounded-xl border border-border bg-card shadow-2xl overflow-hidden',
          'w-[620px] max-h-[680px]',
          'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2',
          'animate-in fade-in-0 zoom-in-95 duration-150',
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-3 shrink-0">
          <div>
            <p className="text-sm font-semibold text-foreground">
              {isAddingFromNode ? 'Add to workflow' : 'Start building'}
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Choose a block, tool, or trigger
            </p>
          </div>
          <Button
            size="icon"
            variant="ghost"
            className="h-7 w-7 shrink-0"
            onClick={dismiss}
            aria-label="Close block picker"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Search */}
        <div className="px-4 pb-3 shrink-0">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              ref={searchInputRef}
              placeholder="Search blocks, tools, triggers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 h-9 text-sm"
            />
          </div>
        </div>

        {/* Tabs for Blocks & Tools vs Triggers */}
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as 'blocks' | 'triggers')}
          className="flex-1 min-h-0 flex flex-col"
        >
          <div className="px-4 shrink-0">
            <TabsList className="w-full grid grid-cols-2 h-9">
              <TabsTrigger value="blocks" className="text-xs">
                Blocks & Tools
              </TabsTrigger>
              <TabsTrigger value="triggers" className="text-xs">
                Triggers
              </TabsTrigger>
            </TabsList>
          </div>

          {/* Blocks & Tools Tab */}
          <TabsContent value="blocks" className="flex-1 min-h-0 overflow-y-auto mt-0">
            <div className="px-4 pb-4 pt-3 space-y-4">
              {blockAndToolItems.length === 0 && (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  {searchQuery ? 'No blocks or tools found' : 'No blocks available'}
                </div>
              )}

              {blockAndToolGroups.map((group, idx) => (
                <div key={`${group.type}-${idx}`}>
                  <h3 className="text-xs font-semibold text-muted-foreground mb-2 px-0.5 uppercase tracking-wide">
                    {getTypeLabel(group.type)}
                  </h3>
                  <div className={getLayoutClass(group.type)}>
                    {group.items.map((item) => (
                      <UnifiedBuildItem
                        key={item.id}
                        item={item}
                        onItemClick={(clickedItem) => {
                          handleItemClick(clickedItem);
                          clearInlineBlockPicker();
                          setSearchQuery('');
                        }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>

          {/* Triggers Tab */}
          <TabsContent value="triggers" className="flex-1 min-h-0 overflow-y-auto mt-0">
            <div className="px-4 pb-4 pt-3 space-y-4">
              {triggerItems.length === 0 && (
                <div className="py-8 text-center text-sm text-muted-foreground">
                  {searchQuery ? 'No triggers found' : 'No triggers available'}
                </div>
              )}

              {triggerGroups.map((group, idx) => (
                <div key={`${group.type}-${idx}`}>
                  <h3 className="text-xs font-semibold text-muted-foreground mb-2 px-0.5 uppercase tracking-wide">
                    {getTypeLabel(group.type)}
                  </h3>
                  <div className={getLayoutClass(group.type)}>
                    {group.items.map((item) => (
                      <UnifiedBuildItem
                        key={item.id}
                        item={item}
                        onItemClick={(clickedItem) => {
                          handleItemClick(clickedItem);
                          clearInlineBlockPicker();
                          setSearchQuery('');
                        }}
                      />
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </>,
    document.body,
  );
}
