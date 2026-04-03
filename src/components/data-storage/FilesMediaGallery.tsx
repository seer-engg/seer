import { useState, useCallback, useMemo } from "react";
import { Trash2, Loader2 } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { filesApi, type UserFile, type FileListParams } from "@/lib/files-api";
import { fileKeys } from "@/lib/query-keys";
import { FileUploadArea } from "./FileUploadArea";
import { usePagination, useFileOperations } from "./gallery/useFileGallery";
import { FileCard, FileRow, EmptyState, GridSkeleton, ListSkeleton } from "./gallery/FileGalleryItems";
import { GalleryToolbar } from "./gallery/GalleryToolbar";
import { BulkDeleteDialog, GalleryPreviewDialog } from "./gallery/GalleryDialogs";
import { GalleryPagination } from "./gallery/GalleryPagination";

const PAGE_SIZE = 24;
type ViewMode = "grid" | "list";

function FileDisplay({ files, viewMode, selectedIds, deletingFileId, onToggleSelect, onPreview, onDownload, onCopyLink, onDelete }: {
  files: UserFile[]; viewMode: ViewMode; selectedIds: Set<string>; deletingFileId: string | null;
  onToggleSelect: (id: string) => void; onPreview: (file: UserFile) => void; onDownload: (id: string) => void; onCopyLink: (id: string) => void; onDelete: (id: string) => void;
}) {
  const Item = viewMode === "grid" ? FileCard : FileRow;
  const cls = viewMode === "grid" ? "grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4" : "space-y-1";
  return (
    <div className={cls}>
      {files.map((file) => (
        <Item key={file.file_id} file={file} isSelected={selectedIds.has(file.file_id)} onToggleSelect={() => onToggleSelect(file.file_id)} onPreview={() => onPreview(file)} onDownload={() => onDownload(file.file_id)} onCopyLink={() => onCopyLink(file.file_id)} onDelete={() => onDelete(file.file_id)} isDeleting={deletingFileId === file.file_id} />
      ))}
    </div>
  );
}

function SelectionBar({ files, selectedIds, totalCount, isPending, onSelectAll, onBulkDelete }: {
  files: UserFile[]; selectedIds: Set<string>; totalCount: number; isPending: boolean; onSelectAll: () => void; onBulkDelete: () => void;
}) {
  return (
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
  );
}

export function FilesMediaGallery() {
  const [viewMode, setViewMode] = useState<ViewMode>("grid");
  const [searchQuery, setSearchQuery] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [sortOption, setSortOption] = useState("created_at:desc");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showUpload, setShowUpload] = useState(false);
  const [previewFile, setPreviewFile] = useState<UserFile | null>(null);
  const { cursor, currentPage, hasPrev, goNext, goPrev, reset } = usePagination();
  const ops = useFileOperations();

  const [sortBy, sortOrder] = sortOption.split(":") as [FileListParams["sort_by"], FileListParams["sort_order"]];
  const queryParams: FileListParams = useMemo(
    () => ({ limit: PAGE_SIZE, cursor: cursor || undefined, sort_by: sortBy, sort_order: sortOrder, filename: searchQuery || undefined, mime_type: typeFilter !== "all" ? typeFilter : undefined }),
    [cursor, sortBy, sortOrder, searchQuery, typeFilter]
  );
  const { data, isLoading, isFetching } = useQuery({
    queryKey: fileKeys.list(queryParams),
    queryFn: () => filesApi.listFiles(queryParams),
    placeholderData: (prev) => prev,
  });

  const files = useMemo(() => data?.files ?? [], [data?.files]);
  const fileIds = useMemo(() => files.map((f) => f.file_id), [files]);

  const resetFilters = useCallback(() => { reset(); setSelectedIds(new Set()); }, [reset]);
  const handleSearchChange = useCallback((v: string) => { setSearchQuery(v); resetFilters(); }, [resetFilters]);
  const handleTypeChange = useCallback((v: string) => { setTypeFilter(v); resetFilters(); }, [resetFilters]);
  const handleSortChange = useCallback((v: string) => { setSortOption(v); resetFilters(); }, [resetFilters]);
  const handleToggleSelect = useCallback((id: string) => setSelectedIds((p) => { const n = new Set(p); if (n.has(id)) n.delete(id); else n.add(id); return n; }), []);
  const handleSelectAll = useCallback(() => setSelectedIds((p) => (p.size === fileIds.length ? new Set() : new Set(fileIds))), [fileIds]);
  const handleBulkDelete = useCallback(() => ops.bulkDeleteMutation.mutate(Array.from(selectedIds), { onSuccess: () => { setSelectedIds(new Set()); setShowDeleteConfirm(false); } }), [ops.bulkDeleteMutation, selectedIds]);

  const content = isLoading
    ? (viewMode === "grid" ? <GridSkeleton /> : <ListSkeleton />)
    : files.length === 0
      ? <EmptyState hasFilters={searchQuery !== "" || typeFilter !== "all"} />
      : <FileDisplay files={files} viewMode={viewMode} selectedIds={selectedIds} deletingFileId={ops.deletingFileId} onToggleSelect={handleToggleSelect} onPreview={setPreviewFile} onDownload={ops.handleDownload} onCopyLink={ops.handleCopyLink} onDelete={ops.handleDelete} />;

  return (
    <>
      <div className="space-y-4">
        {showUpload && <FileUploadArea onUploadComplete={() => {}} />}
        <GalleryToolbar searchQuery={searchQuery} typeFilter={typeFilter} sortOption={sortOption} viewMode={viewMode} onSearchChange={handleSearchChange} onTypeChange={handleTypeChange} onSortChange={handleSortChange} onViewModeChange={setViewMode} onToggleUpload={() => setShowUpload((p) => !p)} />
        <SelectionBar files={files} selectedIds={selectedIds} totalCount={data?.total_count ?? 0} isPending={ops.bulkDeleteMutation.isPending} onSelectAll={handleSelectAll} onBulkDelete={() => setShowDeleteConfirm(true)} />
        {content}
        <GalleryPagination currentPage={currentPage} hasNext={!!data?.next_cursor} hasPrev={hasPrev} isFetching={isFetching} onPrev={goPrev} onNext={() => data?.next_cursor && goNext(data.next_cursor)} />
      </div>
      <BulkDeleteDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm} count={selectedIds.size} isPending={ops.bulkDeleteMutation.isPending} onConfirm={handleBulkDelete} />
      <GalleryPreviewDialog file={previewFile} onClose={() => setPreviewFile(null)} />
    </>
  );
}
