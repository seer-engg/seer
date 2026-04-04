import { ChevronRight } from 'lucide-react';
import type { VariableTreeItem } from '../helpers/variableTreeUtils';
import { parseBreadcrumbs } from '../helpers/variableTreeUtils';

interface VariableAutocompleteDropdownProps {
  visible: boolean;
  items: VariableTreeItem[];
  selectedIndex: number;
  currentPath: string;
  onSelect: (fullPath: string) => void;
  onDrillInto: (item: VariableTreeItem) => void;
  onNavigateTo: (path: string) => void;
  className?: string;
  emptyMessage?: string;
}

export function VariableAutocompleteDropdown({
  visible,
  items,
  selectedIndex,
  currentPath,
  onSelect,
  onDrillInto,
  onNavigateTo,
  className = '',
  emptyMessage = 'No variables found',
}: VariableAutocompleteDropdownProps) {
  if (!visible) {
    return null;
  }

  const breadcrumbs = parseBreadcrumbs(currentPath);

  return (
    <div
      data-testid="variable-autocomplete-dropdown"
      onMouseDown={(e) => e.preventDefault()}
      className={`absolute z-50 mt-1 w-72 rounded-md border bg-popover shadow-md ${className}`}
    >
      {breadcrumbs.length > 0 && (
        <div className="flex items-center gap-0.5 border-b bg-muted/50 px-2 py-1.5 text-xs text-muted-foreground">
          <button
            type="button"
            onClick={() => onNavigateTo('')}
            className="shrink-0 rounded px-1 hover:bg-accent hover:text-accent-foreground"
          >
            root
          </button>
          {breadcrumbs.map((crumb) => (
            <span key={crumb.path} className="flex items-center gap-0.5">
              <ChevronRight className="h-3 w-3 shrink-0 text-muted-foreground/60" />
              <button
                type="button"
                onClick={() => onNavigateTo(crumb.path)}
                className="truncate rounded px-1 hover:bg-accent hover:text-accent-foreground"
              >
                {crumb.label}
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="max-h-60 overflow-auto p-1">
        {items.length === 0 ? (
          <div className="px-2 py-1.5 text-sm text-muted-foreground">{emptyMessage}</div>
        ) : (
          items.map((item, index) => (
            <div
              key={item.fullPath}
              data-testid="variable-suggestion-item"
              onClick={() => onSelect(item.fullPath)}
              className={`flex cursor-pointer items-center justify-between rounded-sm px-2 py-1.5 text-sm hover:bg-accent ${
                index === selectedIndex ? 'bg-accent' : ''
              }`}
            >
              <span className="truncate">{item.label}</span>
              {item.hasChildren && (
                <button
                  type="button"
                  data-testid="variable-drill-in"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDrillInto(item);
                  }}
                  className="ml-1 shrink-0 rounded p-0.5 text-muted-foreground hover:bg-accent-foreground/10 hover:text-foreground"
                >
                  <ChevronRight className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
