import { useCallback, useMemo } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { FileInputField } from './FileInputField';
import type { FileInputValue } from '../../helpers/fileInputSchema';
import type { FileVariable } from '../../helpers/discoverFileVariables';
import type { BaseFieldProps } from './types';

// ============================================================================
// Types
// ============================================================================

export interface FileInputArrayFieldProps extends Omit<BaseFieldProps, 'value' | 'onChange' | 'templateAutocomplete'> {
  value: FileInputValue[];
  onChange: (value: FileInputValue[]) => void;
  /** Available file variables from upstream nodes */
  availableFileVariables?: FileVariable[];
  /** Optional MIME type filter for file picker */
  mimeTypeFilter?: string;
}

// ============================================================================
// Main Component
// ============================================================================

export function FileInputArrayField({
  id,
  value = [],
  onChange,
  placeholder,
  showError,
  availableFileVariables = [],
  mimeTypeFilter,
}: FileInputArrayFieldProps) {
  // Ensure value is always an array (memoized to prevent callback dependency issues)
  const items = useMemo(() => (Array.isArray(value) ? value : []), [value]);

  // Handle adding a new item
  const handleAdd = useCallback(() => {
    onChange([...items, null]);
  }, [items, onChange]);

  // Handle removing an item
  const handleRemove = useCallback(
    (index: number) => {
      const newItems = items.filter((_, i) => i !== index);
      onChange(newItems);
    },
    [items, onChange],
  );

  // Handle changing an item
  const handleItemChange = useCallback(
    (index: number, newValue: FileInputValue) => {
      const newItems = [...items];
      newItems[index] = newValue;
      onChange(newItems);
    },
    [items, onChange],
  );

  return (
    <div className="space-y-3">
      {/* Item List */}
      {items.length > 0 && (
        <div className="space-y-2">
          {items.map((item, index) => (
            <div
              key={index}
              className={cn(
                'relative p-3 rounded-lg border bg-card',
                showError && !item && 'border-destructive/50',
              )}
            >
              {/* Item Number & Remove Button */}
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-muted-foreground">
                  File {index + 1}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => handleRemove(index)}
                  className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>

              {/* File Input Field */}
              <FileInputField
                id={`${id}-${index}`}
                value={item}
                onChange={(newValue) => handleItemChange(index, newValue)}
                placeholder={placeholder}
                showError={showError && !item}
                availableFileVariables={availableFileVariables}
                mimeTypeFilter={mimeTypeFilter}
              />
            </div>
          ))}
        </div>
      )}

      {/* Add Button */}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleAdd}
        className="w-full"
      >
        <Plus className="h-4 w-4 mr-2" />
        Add file
      </Button>

      {/* Empty state hint */}
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground text-center">
          No files added yet. Click "Add file" to attach files.
        </p>
      )}
    </div>
  );
}
