import { File, Folder, FileSpreadsheet, Check, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ResourceItem } from './types';

export interface ResourceListItemProps {
  item: ResourceItem;
  isSelected: boolean;
  onSelect: (item: ResourceItem) => void;
  supportsHierarchy: boolean;
}

function getItemIcon(item: ResourceItem) {
  if (item.icon_url) {
    return <img src={item.icon_url} alt="" className="w-4 h-4" />;
  }

  switch (item.type) {
    case 'folder':
      return <Folder className="w-4 h-4 text-blue-500" />;
    case 'sheet_tab':
      return <FileSpreadsheet className="w-4 h-4 text-green-600" />;
    default:
      if (item.mime_type?.includes('spreadsheet')) {
        return <FileSpreadsheet className="w-4 h-4 text-green-600" />;
      }
      return <File className="w-4 h-4 text-gray-500" />;
  }
}

export function ResourceListItem({
  item,
  isSelected,
  onSelect,
  supportsHierarchy,
}: ResourceListItemProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-3 p-2 rounded-md cursor-pointer hover:bg-accent transition-colors",
        isSelected && "bg-accent"
      )}
      onClick={() => onSelect(item)}
    >
      {getItemIcon(item)}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">
          {item.display_name}
        </p>
        {item.description && (
          <p className="text-xs text-muted-foreground truncate">
            {item.description}
          </p>
        )}
      </div>
      {item.type === 'folder' && item.has_children && supportsHierarchy && (
        <ChevronRight className="h-4 w-4 text-muted-foreground" />
      )}
      {isSelected && (
        <Check className="h-4 w-4 text-primary" />
      )}
    </div>
  );
}
