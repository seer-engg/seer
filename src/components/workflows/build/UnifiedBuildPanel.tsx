import { useMemo, useState, useEffect, useRef } from 'react';
import { Search, X } from 'lucide-react';

import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useUIStore } from '@/stores/uiStore';

import { BUILT_IN_BLOCKS } from '../constants';
import type { BuiltInBlock, Tool } from '../buildtypes';
import type { TriggerListOption } from './TriggerSection';
import { UnifiedBuildItem } from './UnifiedBuildItem';
import { useUnifiedItems } from '@/hooks/useUnifiedItems';
import { useItemSearch } from '@/hooks/useItemSearch';
import { useItemSelection } from '@/hooks/useItemSelection';
import { groupConsecutiveItemsByType } from '@/lib/item-grouping';
import { getLayoutClass, getTypeLabel } from './utils';

interface UnifiedBuildPanelProps {
  tools: Tool[];
  isLoadingTools: boolean;
  onBlockSelect?: (block: { type: string; label: string; config?: Record<string, unknown> }) => void;
  blocks?: BuiltInBlock[];
  selectedWorkflowId?: string | null;
  triggerOptions?: TriggerListOption[];
  isLoadingTriggers?: boolean;
  triggerInfoMessage?: string;
}

// eslint-disable-next-line max-lines-per-function
export function UnifiedBuildPanel({
  tools,
  isLoadingTools,
  onBlockSelect,
  blocks = BUILT_IN_BLOCKS,
  triggerOptions = [],
  isLoadingTriggers = false,
}: UnifiedBuildPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const searchInputRef = useRef<HTMLInputElement>(null);
  const unifiedItems = useUnifiedItems(blocks, triggerOptions, tools);
  const filteredItems = useItemSearch(unifiedItems, searchQuery);
  const itemGroups = useMemo(() => groupConsecutiveItemsByType(filteredItems), [filteredItems]);
  const handleItemClick = useItemSelection(onBlockSelect);
  const isLoading = isLoadingTools || isLoadingTriggers;
  const [highlightActive, setHighlightActive] = useState(false);

  // Pending connection state
  const pendingConnection = useUIStore((state) => state.pendingConnection);
  const setBuildChatPanelCollapsed = useUIStore((state) => state.setBuildChatPanelCollapsed);
  const clearPendingConnection = useUIStore((state) => state.clearPendingConnection);

  // Auto-expand panel when pending connection is active
  useEffect(() => {
    if (pendingConnection.mode !== null) {
      setBuildChatPanelCollapsed(false);
    }
  }, [pendingConnection.mode, setBuildChatPanelCollapsed]);

  useEffect(() => {
    if (pendingConnection.mode !== null) {
      setHighlightActive(true);
      // Focus the search input when pending connection is activated
      searchInputRef.current?.focus();
      const timeout = setTimeout(() => setHighlightActive(false), 500);
      return () => clearTimeout(timeout);
    }
    setHighlightActive(false);
    return undefined;
  }, [pendingConnection.mode, pendingConnection.sourceNodeId, pendingConnection.edgeId, pendingConnection.branch]);

  return (
    <div
      data-testid="build-panel"
      data-collapsed={useUIStore((state) => state.isBuildChatPanelCollapsed) ? 'true' : 'false'}
      className={cn(
        'flex flex-col h-full transition-[border,box-shadow,background-color] duration-200 rounded-md',
        highlightActive && 'border-2 border-seer/60 shadow-[0_0_0_1px_rgba(78,67,252,0.25)] bg-seer/5',
      )}
    >
      {/* Pending connection indicator */}
      {pendingConnection.mode && (
        <div className="px-3 py-2 bg-seer/10 border-b border-seer/20 flex items-center justify-between gap-2">
          <p className="text-xs text-seer flex-1">
            {pendingConnection.mode === 'append'
              ? '✨ Select a block to add after current node'
              : '✨ Select a block to insert between nodes'}
          </p>
          <Button
            size="sm" variant="ghost"
            className="h-6 w-6 p-0 hover:bg-seer/20" onClick={clearPendingConnection} aria-label="Cancel">
            <X className="h-3 w-3" />
          </Button>
        </div>
      )}
      {/* Search bar */}
      <div className="px-3 py-2.5">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            ref={searchInputRef}
            data-testid="build-search"
            placeholder="Search workflow steps, triggers..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8 h-9"
          />
        </div>
      </div>

      {/* Scrollable items list */}
      <ScrollArea className="flex-1">
        <div className="px-3 py-2 space-y-3">
          {isLoading && (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              Loading...
            </div>
          )}

          {!isLoading && filteredItems.length === 0 && (
            <div className="flex items-center justify-center py-8 text-muted-foreground text-sm">
              {searchQuery ? 'No items found' : 'No items available'}
            </div>
          )}

          {!isLoading &&
            itemGroups.map((group, idx) => (
                <div key={`${group.type}-${idx}`}>
                <h3 className="text-xs text-left font-semibold text-muted-foreground mb-2 px-0.5">
                  {getTypeLabel(group.type)}
                </h3>
                <div className={getLayoutClass(group.type)}>
                  {group.items.map((item) => (
                    <UnifiedBuildItem key={item.id} item={item} onItemClick={handleItemClick} />
                  ))}
                </div>
              </div>
            ))}
        </div>
      </ScrollArea>
    </div>
  );
}
