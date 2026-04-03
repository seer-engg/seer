import { useState, useCallback, useMemo } from "react";
import {
  Search,
  Download,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  Loader2,
  Eye,
} from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/utility/use-toast";
import {
  filesApi,
  formatRelativeDate,
  type UserFile,
  type FileListParams,
} from "@/lib/files-api";
import { fileKeys } from "@/lib/query-keys";
import { FileUploadArea } from "./FileUploadArea";
import { FileTypeBadge } from "./FileTypeBadge";
import { FilePreviewDialog } from "./FilePreviewDialog";
import { cn } from "@/lib/utils";

interface FilesManagementDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const PAGE_SIZE = 20;

const SORT_OPTIONS = [
  { value: "created_at:desc", label: "Newest first" },
  { value: "created_at:asc", label: "Oldest first" },
  { value: "size_bytes:desc", label: "Largest first" },
  { value: "size_bytes:asc", label: "Smallest first" },
  { value: "filename:asc", label: "Name A-Z" },
  { value: "filename:desc", label: "Name Z-A" },
];

const TYPE_FILTERS = [
  { value: "all", label: "All types" },
  { value: "application/pdf", label: "PDF" },
  { value: "image/*", label: "Images" },
  { value: "text/*", label: "Text" },
  { value: "application/json", label: "JSON" },
];

// Extracted components to reduce main function complexity

function FileRow({
  file,
  isSelected,
  onToggleSelect,
  onPreview,
  onDownload,
  onDelete,
  isDeleting,
}: {
  file: UserFile;
  isSelected: boolean;
  onToggleSelect: () => void;
  onPreview: () => void;
  onDownload: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}) {
  const sourceText = useMemo(() => {
    if (file.workflow_name) return `From: ${file.workflow_name}`;
    if (file.source_tool === "user_upload") return "User upload";
    if (file.source_tool) return `Tool: ${file.source_tool}`;
    return null;
  }, [file.workflow_name, file.source_tool]);

  return (
    <div
      className={cn(
        "flex items-center gap-3 py-2.5 px-3 rounded-lg transition-colors",
        isSelected ? "bg-primary/5" : "hover:bg-muted/50"
      )}
    >
      <Checkbox checked={isSelected} onCheckedChange={onToggleSelect} />
      <div className="flex-1 min-w-0 flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{file.filename}</p>
          {sourceText && (
            <p className="text-xs text-muted-foreground truncate">{sourceText}</p>
          )}
        </div>
        <FileTypeBadge mimeType={file.mime_type} className="flex-shrink-0" />
        <span className="text-xs text-muted-foreground w-16 text-right flex-shrink-0">
          {file.size_human}
        </span>
        <span className="text-xs text-muted-foreground w-20 text-right flex-shrink-0">
          {formatRelativeDate(file.created_at)}
        </span>
      </div>
      <div className="flex items-center gap-1 flex-shrink-0">
        <Button
          variant="ghost"
          size="sm"
          onClick={onPreview}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
          title="Preview"
        >
          <Eye className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDownload}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
          title="Download"
        >
          <Download className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onDelete}
          disabled={isDeleting}
          className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
          title="Delete"
        >
          {isDeleting ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Trash2 className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
    </div>
  );
}

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  return (
    <div className="py-12 text-center">
      <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-3">
        <Search className="h-6 w-6 text-muted-foreground/50" />
      </div>
      <p className="text-sm text-muted-foreground">
        {hasFilters ? "No files match your filters" : "No files yet"}
      </p>
      <p className="text-xs text-muted-foreground mt-1">
        {hasFilters ? "Try adjusting your search or filters" : "Upload files to get started"}
      </p>
    </div>
  );
}

function LoadingSkeleton() {
  return (
    <div className="space-y-2">
      {[1, 2, 3, 4, 5].map((i) => (
        <div key={i} className="flex items-center gap-3 py-2.5 px-3">
          <div className="w-4 h-4 bg-muted rounded animate-pulse" />
          <div className="flex-1 space-y-1">
            <div className="h-4 bg-muted rounded w-1/3 animate-pulse" />
            <div className="h-3 bg-muted rounded w-1/4 animate-pulse" />
          </div>
          <div className="h-5 w-12 bg-muted rounded animate-pulse" />
          <div className="h-4 w-16 bg-muted rounded animate-pulse" />
        </div>
      ))}
    </div>
  );
}

function FileFilters({
  searchQuery,
  typeFilter,
  sortOption,
  onSearchChange,
  onTypeChange,
  onSortChange,
}: {
  searchQuery: string;
  typeFilter: string;
  sortOption: string;
  onSearchChange: (value: string) => void;
  onTypeChange: (value: string) => void;
  onSortChange: (value: string) => void;
}) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <div className="relative flex-1 min-w-[200px]">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search files..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-8 h-9"
        />
      </div>
      <Select value={typeFilter} onValueChange={onTypeChange}>
        <SelectTrigger className="w-[130px] h-9">
          <SelectValue placeholder="Type" />
        </SelectTrigger>
        <SelectContent>
          {TYPE_FILTERS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={sortOption} onValueChange={onSortChange}>
        <SelectTrigger className="w-[140px] h-9">
          <ArrowUpDown className="h-3.5 w-3.5 mr-1.5" />
          <SelectValue placeholder="Sort" />
        </SelectTrigger>
        <SelectContent>
          {SORT_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

function FilePagination({
  currentPage,
  hasNextPage,
  hasPrevPage,
  isFetching,
  onPrev,
  onNext,
}: {
  currentPage: number;
  hasNextPage: boolean;
  hasPrevPage: boolean;
  isFetching: boolean;
  onPrev: () => void;
  onNext: () => void;
}) {
  if (!hasNextPage && !hasPrevPage) return null;

  return (
    <div className="flex items-center justify-between pt-2 border-t">
      <Button
        variant="outline"
        size="sm"
        onClick={onPrev}
        disabled={!hasPrevPage || isFetching}
        className="h-8"
      >
        <ChevronLeft className="h-4 w-4 mr-1" />
        Previous
      </Button>
      <span className="text-xs text-muted-foreground">Page {currentPage}</span>
      <Button
        variant="outline"
        size="sm"
        onClick={onNext}
        disabled={!hasNextPage || isFetching}
        className="h-8"
      >
        Next
        <ChevronRight className="h-4 w-4 ml-1" />
      </Button>
    </div>
  );
}

function BulkDeleteConfirmDialog({
  open,
  onOpenChange,
  count,
  isPending,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  count: number;
  isPending: boolean;
  onConfirm: () => void;
}) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {count} file(s)?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone. The selected files will be permanently deleted.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            disabled={isPending}
          >
            {isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
            Delete Files
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

// Custom hook to manage file operations
function useFileOperations() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [deletingFileId, setDeletingFileId] = useState<string | null>(null);

  const deleteMutation = useMutation({
    mutationFn: filesApi.deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: fileKeys.all });
      toast({ title: "File deleted", description: "The file has been removed." });
      setDeletingFileId(null);
    },
    onError: (error: Error) => {
      toast({ title: "Failed to delete file", description: error.message, variant: "destructive" });
      setDeletingFileId(null);
    },
  });

  const bulkDeleteMutation = useMutation({
    mutationFn: filesApi.bulkDeleteFiles,
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: fileKeys.all });
      toast({ title: "Files deleted", description: `${result.deleted_count} file(s) removed.` });
    },
    onError: (error: Error) => {
      toast({ title: "Failed to delete files", description: error.message, variant: "destructive" });
    },
  });

  const handleDownload = useCallback(async (fileId: string) => {
    try {
      const response = await filesApi.getDownloadUrl(fileId);
      window.open(response.download_url, "_blank");
    } catch (error) {
      toast({ title: "Download failed", description: error instanceof Error ? error.message : "Failed to get download URL", variant: "destructive" });
    }
  }, [toast]);

  const handleDelete = useCallback((fileId: string) => {
    setDeletingFileId(fileId);
    deleteMutation.mutate(fileId);
  }, [deleteMutation]);

  return { deletingFileId, bulkDeleteMutation, handleDownload, handleDelete };
}

// File list with selection header
function FileListSection({
  files, totalCount, selectedIds, isLoading, hasFilters, deletingFileId, isPending,
  onToggleSelect, onSelectAll, onPreview, onDownload, onDelete, onBulkDelete,
}: {
  files: UserFile[]; totalCount: number; selectedIds: Set<string>; isLoading: boolean; hasFilters: boolean; deletingFileId: string | null; isPending: boolean;
  onToggleSelect: (id: string) => void; onSelectAll: () => void; onPreview: (file: UserFile) => void; onDownload: (id: string) => void; onDelete: (id: string) => void; onBulkDelete: () => void;
}) {
  return (
    <>
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-2">
          <Checkbox checked={files.length > 0 && selectedIds.size === files.length} onCheckedChange={onSelectAll} disabled={files.length === 0} />
          <span>{selectedIds.size > 0 ? `${selectedIds.size} selected` : `${totalCount} file${totalCount !== 1 ? "s" : ""}`}</span>
        </div>
        {selectedIds.size > 0 && (
          <Button variant="destructive" size="sm" className="h-7 text-xs" onClick={onBulkDelete} disabled={isPending}>
            {isPending ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Trash2 className="h-3 w-3 mr-1" />}
            Delete ({selectedIds.size})
          </Button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto -mx-6 px-6 min-h-0">
        {isLoading ? <LoadingSkeleton /> : files.length === 0 ? <EmptyState hasFilters={hasFilters} /> : (
          <div className="space-y-1">
            {files.map((file) => (
              <FileRow key={file.file_id} file={file} isSelected={selectedIds.has(file.file_id)} onToggleSelect={() => onToggleSelect(file.file_id)} onPreview={() => onPreview(file)} onDownload={() => onDownload(file.file_id)} onDelete={() => onDelete(file.file_id)} isDeleting={deletingFileId === file.file_id} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}

// Hook for pagination state
function usePagination() {
  const [cursor, setCursor] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const goNext = useCallback((nextCursor: string) => { setHistory((p) => [...p, cursor || ""]); setCursor(nextCursor); }, [cursor]);
  const goPrev = useCallback(() => { const h = [...history]; const prev = h.pop() || null; setHistory(h); setCursor(prev === "" ? null : prev); }, [history]);
  const reset = useCallback(() => { setCursor(null); setHistory([]); }, []);
  return { cursor, currentPage: history.length + 1, hasPrev: history.length > 0, goNext, goPrev, reset };
}

export function FilesManagementDialog({ open, onOpenChange }: FilesManagementDialogProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [sortOption, setSortOption] = useState("created_at:desc");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [previewFile, setPreviewFile] = useState<UserFile | null>(null);
  const { cursor, currentPage, hasPrev, goNext, goPrev, reset } = usePagination();
  const { deletingFileId, bulkDeleteMutation, handleDownload, handleDelete } = useFileOperations();

  const handlePreview = useCallback((file: UserFile) => {
    setPreviewFile(file);
  }, []);

  const [sortBy, sortOrder] = sortOption.split(":") as [FileListParams["sort_by"], FileListParams["sort_order"]];
  const queryParams: FileListParams = useMemo(() => ({ limit: PAGE_SIZE, cursor: cursor || undefined, sort_by: sortBy, sort_order: sortOrder, filename: searchQuery || undefined, mime_type: typeFilter !== "all" ? typeFilter : undefined }), [cursor, sortBy, sortOrder, searchQuery, typeFilter]);
  const { data, isLoading, isFetching } = useQuery({
    queryKey: fileKeys.list(queryParams),
    queryFn: () => filesApi.listFiles(queryParams),
    enabled: open,
    placeholderData: (prev) => prev,
  });

  const files = useMemo(() => data?.files ?? [], [data?.files]);
  const fileIds = useMemo(() => files.map((f) => f.file_id), [files]);
  const handleFilterChange = useCallback((setter: (v: string) => void, v: string) => { setter(v); reset(); setSelectedIds(new Set()); }, [reset]);
  const handleToggleSelect = useCallback((id: string) => setSelectedIds((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n; }), []);
  const handleSelectAll = useCallback(() => setSelectedIds((p) => p.size === fileIds.length ? new Set() : new Set(fileIds)), [fileIds]);
  const handleBulkDelete = useCallback(() => bulkDeleteMutation.mutate(Array.from(selectedIds), { onSuccess: () => { setSelectedIds(new Set()); setShowDeleteConfirm(false); } }), [bulkDeleteMutation, selectedIds]);

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-3xl max-h-[85vh] flex flex-col">
          <DialogHeader><DialogTitle>Manage Files</DialogTitle><DialogDescription>Upload, download, and manage your workflow files</DialogDescription></DialogHeader>
          <div className="flex-1 overflow-hidden flex flex-col space-y-4">
            <FileUploadArea />
            <FileFilters searchQuery={searchQuery} typeFilter={typeFilter} sortOption={sortOption} onSearchChange={(v) => handleFilterChange(setSearchQuery, v)} onTypeChange={(v) => handleFilterChange(setTypeFilter, v)} onSortChange={(v) => handleFilterChange(setSortOption, v)} />
            <FileListSection files={files} totalCount={data?.total_count ?? 0} selectedIds={selectedIds} isLoading={isLoading} hasFilters={searchQuery !== "" || typeFilter !== "all"} deletingFileId={deletingFileId} isPending={bulkDeleteMutation.isPending} onToggleSelect={handleToggleSelect} onSelectAll={handleSelectAll} onPreview={handlePreview} onDownload={handleDownload} onDelete={handleDelete} onBulkDelete={() => setShowDeleteConfirm(true)} />
            <FilePagination currentPage={currentPage} hasNextPage={!!data?.next_cursor} hasPrevPage={hasPrev} isFetching={isFetching} onPrev={goPrev} onNext={() => data?.next_cursor && goNext(data.next_cursor)} />
          </div>
        </DialogContent>
      </Dialog>
      <BulkDeleteConfirmDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm} count={selectedIds.size} isPending={bulkDeleteMutation.isPending} onConfirm={handleBulkDelete} />
      <FilePreviewDialog
        open={previewFile !== null}
        onOpenChange={(open) => !open && setPreviewFile(null)}
        file={previewFile}
      />
    </>
  );
}

export type { FilesManagementDialogProps };
