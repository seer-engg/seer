/**
 * Individual trigger event item in the picker list.
 * Displays title, subtitle, and preview with selection state.
 * Supports expanding to show the full envelope via JsonTreeView.
 */

import { useState } from 'react';
import { Mail, MessageSquare, Webhook, CalendarDays, Check, ChevronRight, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { JsonTreeView } from '@/components/workflows/executions/JsonTreeView';
import { TRIGGER_DISPLAY_CONFIG } from './types';
import type { TriggerEventListItemProps } from './types';

function getTriggerIcon(triggerKey: string) {
  const config = TRIGGER_DISPLAY_CONFIG[triggerKey];
  const iconClass = 'w-4 h-4';

  switch (config?.icon) {
    case 'mail':
      return <Mail className={cn(iconClass, 'text-red-500')} />;
    case 'message-square':
      return <MessageSquare className={cn(iconClass, 'text-indigo-500')} />;
    case 'calendar':
      return <CalendarDays className={cn(iconClass, 'text-sky-500')} />;
    case 'zap':
    default:
      return <Webhook className={cn(iconClass, 'text-gray-500')} />;
  }
}

export function TriggerEventListItem({
  item,
  isSelected,
  onSelect,
  triggerKey,
}: TriggerEventListItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleToggleExpand = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsExpanded((prev) => !prev);
  };

  return (
    <div data-testid="trigger-event-item">
      <div
        className={cn(
          'flex items-start gap-3 p-3 rounded-md cursor-pointer hover:bg-accent transition-colors',
          isSelected && 'bg-accent ring-1 ring-primary',
        )}
        onClick={() => onSelect(item)}
      >
        <div className="flex-shrink-0 mt-0.5">{getTriggerIcon(triggerKey)}</div>
        <div className="flex-1 min-w-0 space-y-0.5">
          <p className="text-sm font-medium truncate">{item.display_title}</p>
          {item.display_subtitle && (
            <p className="text-xs text-muted-foreground">{item.display_subtitle}</p>
          )}
          {item.preview && (
            <p className="text-xs text-muted-foreground/70 line-clamp-2">{item.preview}</p>
          )}
        </div>
        {isSelected && <Check className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />}
        <button
          type="button"
          onClick={handleToggleExpand}
          className="flex-shrink-0 mt-0.5 p-1 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
          aria-label={isExpanded ? 'Collapse envelope' : 'Inspect envelope'}
        >
          {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
      </div>
      {isExpanded && item.envelope && (
        <div className="mx-3 mb-2 p-3 rounded-md bg-muted/50 border border-border/50 max-h-[300px] overflow-auto">
          <p className="text-xs font-medium text-muted-foreground mb-2">Envelope Data</p>
          <JsonTreeView data={item.envelope} />
        </div>
      )}
    </div>
  );
}

export type { TriggerEventListItemProps };
