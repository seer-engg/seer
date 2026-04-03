/**
 * Project Dropdown Component
 *
 * Lightweight dropdown wrapper for Supabase project selection.
 * Reuses the ResourcePicker data fetching logic but renders as a dropdown
 * instead of a modal dialog for faster inline selection.
 */
import { useState, useMemo, useEffect } from 'react';
import { ChevronDown, Loader2, Search } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';
import { useResourceFetch } from '../../../dialogs/ResourcePicker/hooks/useResourceFetch';
import {
  normalizeEndpoint,
  buildBaseEndpoint,
} from '../../../dialogs/ResourcePicker/resourcePickerHelpers';

const SUPABASE_PICKER_CONFIG = {
  resource_type: 'supabase_binding',
  endpoint: '/api/integrations/supabase/resources/bindings',
  display_field: 'display_name',
  value_field: 'id',
  search_enabled: true,
} as const;

interface ProjectDropdownProps {
  value: string | number;
  onChange: (value: string | number, label?: string) => void;
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

function ProjectDropdownContent({
  open,
  searchQuery,
  onSearchChange,
  resourceFetch,
  value,
  onSelect,
}: {
  open: boolean;
  searchQuery: string;
  onSearchChange: (query: string) => void;
  resourceFetch: ReturnType<typeof useResourceFetch>;
  value: string | number;
  onSelect: (itemValue: string, itemLabel: string) => void;
}) {
  return (
    <DropdownMenuContent
      className="w-[var(--radix-dropdown-menu-trigger-width)]"
      align="start"
    >
      {/* Search input */}
      <div className="px-2 py-1.5">
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search projects..."
            className="pl-8 h-8 text-sm"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            autoFocus={open}
          />
        </div>
      </div>

      <DropdownMenuSeparator />

      {/* Loading state */}
      {resourceFetch.isLoading && (
        <div className="px-2 py-6 text-center text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 inline-block animate-spin mr-2" />
          Loading projects...
        </div>
      )}

      {/* Error state */}
      {resourceFetch.isError && (
        <div className="px-2 py-3 text-sm text-destructive">
          Failed to load projects
        </div>
      )}

      {/* Items */}
      {!resourceFetch.isLoading && !resourceFetch.isError && resourceFetch.items.length > 0 && (
        <>
          {resourceFetch.items.map((item) => (
            <DropdownMenuItem
              key={item.id}
              onSelect={() => onSelect(String(item.id), item.display_name)}
              className={cn(
                'cursor-pointer',
                String(item.id) === String(value) && 'bg-accent',
              )}
            >
              <span className="text-sm">{item.display_name}</span>
            </DropdownMenuItem>
          ))}
        </>
      )}

      {/* Empty state */}
      {!resourceFetch.isLoading && !resourceFetch.isError && resourceFetch.items.length === 0 && (
        <div className="px-2 py-6 text-center text-sm text-muted-foreground">
          No projects found
        </div>
      )}
    </DropdownMenuContent>
  );
}

export function ProjectDropdown({
  value,
  onChange,
  placeholder = 'Select project',
  disabled = false,
  className,
}: ProjectDropdownProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [displayValue, setDisplayValue] = useState<string>('');

  const normalizedEndpoint = useMemo(
    () => normalizeEndpoint(SUPABASE_PICKER_CONFIG.endpoint),
    [],
  );

  const baseEndpoint = useMemo(
    () => buildBaseEndpoint(normalizedEndpoint, undefined, SUPABASE_PICKER_CONFIG.resource_type),
    [normalizedEndpoint],
  );

  const resourceFetch = useResourceFetch({
    baseEndpoint,
    resourceType: SUPABASE_PICKER_CONFIG.resource_type,
    searchQuery,
    valueField: SUPABASE_PICKER_CONFIG.value_field || 'id',
    displayField: SUPABASE_PICKER_CONFIG.display_field || 'display_name',
    normalizedEndpoint,
    searchEnabled: SUPABASE_PICKER_CONFIG.search_enabled,
    open,
    hasMissingDependency: false,
  });

  // Update display value when selected value changes
  useEffect(() => {
    if (value && resourceFetch.items.length > 0) {
      const selectedItem = resourceFetch.items.find(
        (item) => String(item.id) === String(value),
      );
      if (selectedItem) {
        setDisplayValue(selectedItem.display_name);
      }
    } else if (!value) {
      setDisplayValue('');
    }
  }, [value, resourceFetch.items]);

  const handleSelect = (itemValue: string, itemLabel: string) => {
    onChange(itemValue, itemLabel);
    setDisplayValue(itemLabel);
    setOpen(false);
    setSearchQuery('');
  };

  const handleOpenChange = (newOpen: boolean) => {
    setOpen(newOpen);
    // Refetch projects whenever dropdown is opened
    if (newOpen) {
      resourceFetch.refetch();
    }
  };

  return (
    <DropdownMenu open={open} onOpenChange={handleOpenChange}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="outline"
          className={cn(
            'w-full justify-between text-left font-normal',
            !displayValue && 'text-muted-foreground',
            className,
          )}
          disabled={disabled}
        >
          <span className="truncate">{displayValue || placeholder}</span>
          <ChevronDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </DropdownMenuTrigger>

      <ProjectDropdownContent
        open={open}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        resourceFetch={resourceFetch}
        value={value}
        onSelect={handleSelect}
      />
    </DropdownMenu>
  );
}
