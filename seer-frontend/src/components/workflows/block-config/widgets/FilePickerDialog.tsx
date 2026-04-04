import { useState, useCallback, useMemo } from 'react';
import { Search, ChevronLeft, ChevronRight, Loader2, File, Check } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { filesApi, formatRelativeDate, type UserFile, type FileListParams } from '@/lib/files-api';
import { fileKeys } from '@/lib/query-keys';
import { FileTypeBadge } from '@/components/data-storage/FileTypeBadge';
import { cn } from '@/lib/utils';

// ============================================================================
// Types & Constants
// ============================================================================

export interface FilePickerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (file: UserFile) => void;
  selectedFileId?: string;
  mimeTypeFilter?: string;
  title?: string;
  description?: string;
}

const PAGE_SIZE = 15;
const TYPE_FILTERS = [
  { value: 'all', label: 'All types' },
  { value: 'application/pdf', label: 'PDF' },
  { value: 'image/*', label: 'Images' },
  { value: 'text/*', label: 'Text' },
  { value: 'application/json', label: 'JSON' },
];

// ============================================================================
// Subcomponents
// ============================================================================

function SelectableFileRow({ file, isSelected, onSelect }: { file: UserFile; isSelected: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full flex items-center gap-3 py-2.5 px-3 rounded-lg transition-colors text-left',
        isSelected ? 'bg-primary/10 border border-primary/30' : 'hover:bg-muted/50 border border-transparent',
      )}
    >
      <div className={cn('w-5 h-5 rounded-full border-2 flex items-center justify-center flex-shrink-0', isSelected ? 'border-primary bg-primary' : 'border-muted-foreground/30')}>
        {isSelected && <Check className="h-3 w-3 text-primary-foreground" />}
      </div>
      <div className="flex-1 min-w-0 flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{file.filename}</p>
          {file.workflow_name && <p className="text-xs text-muted-foreground truncate">From: {file.workflow_name}</p>}
        </div>
        <FileTypeBadge mimeType={file.mime_type} className="flex-shrink-0" />
        <span className="text-xs text-muted-foreground w-16 text-right flex-shrink-0">{file.size_human}</span>
        <span className="text-xs text-muted-foreground w-20 text-right flex-shrink-0">{formatRelativeDate(file.created_at)}</span>
      </div>
    </button>
  );
}

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  return (
    <div className="py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-3">
        <File className="h-6 w-6 text-muted-foreground/50" />
      </div>
      <p className="text-sm text-muted-foreground">{hasFilters ? 'No files match your filters' : 'No files in storage'}</p>
      <p className="text-xs text-muted-foreground mt-1">{hasFilters ? 'Try adjusting your search' : 'Upload files in Settings'}</p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center gap-3 py-2.5 px-3">
          <div className="w-5 h-5 bg-muted rounded-full animate-pulse" />
          <div className="flex-1 space-y-1">
            <div className="h-4 bg-muted rounded w-1/3 animate-pulse" />
            <div className="h-3 bg-muted rounded w-1/4 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}

function FilePickerFilters({ searchQuery, onSearchChange, typeFilter, onTypeChange, showTypeFilter }: {
  searchQuery: string; onSearchChange: (v: string) => void;
  typeFilter: string; onTypeChange: (v: string) => void;
  showTypeFilter: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative flex-1">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input placeholder="Search files..." value={searchQuery} onChange={(e) => onSearchChange(e.target.value)} className="pl-8 h-9" />
      </div>
      {showTypeFilter && (
        <Select value={typeFilter} onValueChange={onTypeChange}>
          <SelectTrigger className="w-[130px] h-9"><SelectValue placeholder="Type" /></SelectTrigger>
          <SelectContent>
            {TYPE_FILTERS.map((opt) => <SelectItem key={opt.value} value={opt.value}>{opt.label}</SelectItem>)}
          </SelectContent>
        </Select>
      )}
    </div>
  );
}

function FilePickerPagination({ hasPrev, hasNext, currentPage, isFetching, onPrev, onNext }: {
  hasPrev: boolean; hasNext: boolean; currentPage: number; isFetching: boolean;
  onPrev: () => void; onNext: () => void;
}) {
  if (!hasPrev && !hasNext) return null;
  return (
    <div className="flex items-center justify-between pt-2 border-t">
      <Button variant="outline" size="sm" onClick={onPrev} disabled={!hasPrev || isFetching} className="h-8">
        <ChevronLeft className="h-4 w-4 mr-1" />Previous
      </Button>
      <span className="text-xs text-muted-foreground">Page {currentPage}</span>
      <Button variant="outline" size="sm" onClick={onNext} disabled={!hasNext || isFetching} className="h-8">
        Next<ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  );
}

// ============================================================================
// Hooks
// ============================================================================

function usePagination() {
  const [cursor, setCursor] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const goNext = useCallback((nextCursor: string) => { setHistory((p) => [...p, cursor || '']); setCursor(nextCursor); }, [cursor]);
  const goPrev = useCallback(() => { const h = [...history]; setCursor(h.pop() || null); setHistory(h); }, [history]);
  const reset = useCallback(() => { setCursor(null); setHistory([]); }, []);
  return { cursor, currentPage: history.length + 1, hasPrev: history.length > 0, goNext, goPrev, reset };
}

function useFilePickerState(open: boolean, selectedFileId?: string, mimeTypeFilter?: string) {
  const [searchQuery, setSearchQuery] = useState('');
  const [typeFilter, setTypeFilter] = useState(mimeTypeFilter || 'all');
  const [localSelectedId, setLocalSelectedId] = useState<string | null>(selectedFileId || null);
  const { cursor, currentPage, hasPrev, goNext, goPrev, reset } = usePagination();

  const queryParams: FileListParams = useMemo(() => ({
    limit: PAGE_SIZE, cursor: cursor || undefined, sort_by: 'created_at', sort_order: 'desc',
    filename: searchQuery || undefined, mime_type: typeFilter !== 'all' ? typeFilter : undefined,
  }), [cursor, searchQuery, typeFilter]);

  const { data, isLoading, isFetching } = useQuery({
    queryKey: fileKeys.pickerList(queryParams),
    queryFn: () => filesApi.listFiles(queryParams),
    enabled: open,
    placeholderData: (prev) => prev,
  });

  const files = useMemo(() => data?.files ?? [], [data?.files]);
  const selectedFile = useMemo(() => files.find((f) => f.file_id === localSelectedId), [files, localSelectedId]);
  const hasFilters = searchQuery !== '' || (typeFilter !== 'all' && typeFilter !== mimeTypeFilter);

  const handleFilterChange = useCallback((setter: (v: string) => void, value: string) => { setter(value); reset(); }, [reset]);

  return {
    searchQuery, setSearchQuery, typeFilter, setTypeFilter, localSelectedId, setLocalSelectedId,
    files, selectedFile, isLoading, isFetching, hasFilters, currentPage, hasPrev,
    hasNext: !!data?.next_cursor, handleFilterChange,
    handleGoNext: () => data?.next_cursor && goNext(data.next_cursor), handleGoPrev: goPrev,
  };
}

// ============================================================================
// Main Component
// ============================================================================

export function FilePickerDialog({
  open, onOpenChange, onSelect, selectedFileId, mimeTypeFilter,
  title = 'Select File', description = 'Choose a file from your storage',
}: FilePickerDialogProps) {
  const state = useFilePickerState(open, selectedFileId, mimeTypeFilter);

  const handleOpenChange = useCallback((newOpen: boolean) => {
    if (newOpen) state.setLocalSelectedId(selectedFileId || null);
    onOpenChange(newOpen);
  }, [onOpenChange, selectedFileId, state]);

  const handleConfirm = useCallback(() => {
    if (state.selectedFile) { onSelect(state.selectedFile); onOpenChange(false); }
  }, [state.selectedFile, onSelect, onOpenChange]);

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader><DialogTitle>{title}</DialogTitle><DialogDescription>{description}</DialogDescription></DialogHeader>

        <FilePickerFilters
          searchQuery={state.searchQuery}
          onSearchChange={(v) => state.handleFilterChange(state.setSearchQuery, v)}
          typeFilter={state.typeFilter}
          onTypeChange={(v) => state.handleFilterChange(state.setTypeFilter, v)}
          showTypeFilter={!mimeTypeFilter}
        />

        <div className="flex-1 overflow-y-auto min-h-[300px] max-h-[400px] -mx-6 px-6">
          {state.isLoading ? <LoadingSkeleton /> : state.files.length === 0 ? <EmptyState hasFilters={state.hasFilters} /> : (
            <div className="space-y-1">
              {state.files.map((file) => (
                <SelectableFileRow key={file.file_id} file={file} isSelected={state.localSelectedId === file.file_id} onSelect={() => state.setLocalSelectedId(file.file_id)} />
              ))}
            </div>
          )}
        </div>

        <FilePickerPagination
          hasPrev={state.hasPrev} hasNext={state.hasNext} currentPage={state.currentPage}
          isFetching={state.isFetching} onPrev={state.handleGoPrev} onNext={state.handleGoNext}
        />

        <DialogFooter className="flex-row items-center justify-between sm:justify-between border-t pt-4">
          <div className="text-sm text-muted-foreground">
            {state.selectedFile ? <span>Selected: <span className="font-medium text-foreground">{state.selectedFile.filename}</span> ({state.selectedFile.size_human})</span> : <span>No file selected</span>}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button onClick={handleConfirm} disabled={!state.selectedFile}>
              {state.isFetching && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}Select
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
