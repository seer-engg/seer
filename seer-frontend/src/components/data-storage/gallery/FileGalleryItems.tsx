import { useState, useEffect } from "react";
import * as pdfjs from "pdfjs-dist";
import {
  Download,
  Trash2,
  Loader2,
  Eye,
  Link2,
  FolderOpen,
} from "lucide-react";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  filesApi,
  formatRelativeDate,
  getFileTypeCategory,
  getPreviewCategory,
  type UserFile,
  type PreviewCategory,
} from "@/lib/files-api";
import { backendApiClient } from "@/lib/api-client";
import { FileTypeBadge, FileTypeIcon } from "../FileTypeBadge";
import { cn } from "@/lib/utils";

export interface FileItemProps {
  file: UserFile;
  isSelected: boolean;
  onToggleSelect: () => void;
  onPreview: () => void;
  onDownload: () => void;
  onCopyLink: () => void;
  onDelete: () => void;
  isDeleting: boolean;
}

// ---------------------------------------------------------------------------
// useTextPreview — lightweight fetch for card thumbnail text content
// ---------------------------------------------------------------------------

const PREVIEW_CATEGORIES = new Set<PreviewCategory>(["text", "data"]);
const PREVIEW_MAX_BYTES = 1024;

function shouldFetchPreview(mimeType: string): boolean {
  const cat = getPreviewCategory(mimeType);
  return PREVIEW_CATEGORIES.has(cat);
}

const previewCache = new Map<string, string>();

function useTextPreview(fileId: string, mimeType: string) {
  const enabled = shouldFetchPreview(mimeType);
  const [data, setData] = useState<string | null>(
    enabled ? previewCache.get(fileId) ?? null : null,
  );
  const [isLoading, setIsLoading] = useState(enabled && !previewCache.has(fileId));

  useEffect(() => {
    if (!enabled || previewCache.has(fileId)) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await backendApiClient.stream(filesApi.getContentUrl(fileId));
        if (!res.ok) throw new Error("fetch failed");
        const reader = res.body?.getReader();
        if (!reader) return;

        const chunks: Uint8Array[] = [];
        let total = 0;
        while (total < PREVIEW_MAX_BYTES) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          total += value.length;
        }
        reader.cancel();

        const decoded = chunks
          .map((c) => new TextDecoder().decode(c, { stream: true }))
          .join("")
          .slice(0, PREVIEW_MAX_BYTES);

        previewCache.set(fileId, decoded);
        if (!cancelled) {
          setData(decoded);
          setIsLoading(false);
        }
      } catch {
        if (!cancelled) setIsLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [fileId, enabled]);

  return { data, isLoading };
}

// ---------------------------------------------------------------------------
// useImageUrl — authenticated blob URL for <img> tags
// ---------------------------------------------------------------------------

const imageUrlCache = new Map<string, string>();

function useImageUrl(fileId: string, enabled: boolean) {
  const [url, setUrl] = useState<string | null>(
    enabled ? imageUrlCache.get(fileId) ?? null : null,
  );

  useEffect(() => {
    if (!enabled || imageUrlCache.has(fileId)) return;
    let cancelled = false;

    (async () => {
      try {
        const res = await backendApiClient.stream(filesApi.getContentUrl(fileId));
        const blob = await res.blob();
        const objectUrl = URL.createObjectURL(blob);
        imageUrlCache.set(fileId, objectUrl);
        if (!cancelled) setUrl(objectUrl);
      } catch {
        // fall through — imgError handler in FileThumbnail will show fallback
      }
    })();

    return () => { cancelled = true; };
  }, [fileId, enabled]);

  return url;
}

// ---------------------------------------------------------------------------
// Tiny sub-previews for card thumbnails
// ---------------------------------------------------------------------------

function TextThumbnail({ text }: { text: string }) {
  return (
    <div className="relative w-full h-full bg-muted/20 p-2 overflow-hidden">
      <pre className="text-[8px] leading-[11px] text-muted-foreground font-mono whitespace-pre-wrap break-all">
        {text.slice(0, 500)}
      </pre>
      <div className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-card to-transparent" />
    </div>
  );
}

function JsonThumbnail({ text }: { text: string }) {
  // Basic syntax coloring: keys vs values
  const lines = text.slice(0, 500).split("\n").slice(0, 20);
  return (
    <div className="relative w-full h-full bg-muted/20 p-2 overflow-hidden">
      <pre className="text-[8px] leading-[11px] font-mono whitespace-pre-wrap break-all">
        {lines.map((line, i) => (
          <span key={i}>
            {line.split(/("(?:[^"\\]|\\.)*"\s*):/).map((part, j) =>
              j % 2 === 1 ? (
                <span key={j} className="text-blue-500 dark:text-blue-400">{part}:</span>
              ) : (
                <span key={j} className="text-muted-foreground">{part}</span>
              ),
            )}
            {"\n"}
          </span>
        ))}
      </pre>
      <div className="absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-card to-transparent" />
    </div>
  );
}

function CsvThumbnail({ text }: { text: string }) {
  const lines = text.split(/\r?\n/).filter(Boolean).slice(0, 5);
  const rows = lines.map((l) => l.split(",").map((c) => c.trim().replace(/^"|"$/g, "")));

  return (
    <div className="relative w-full h-full bg-muted/20 p-1.5 overflow-hidden">
      <table className="w-full text-[7px] leading-[10px] font-mono border-collapse">
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={ri === 0 ? "font-semibold text-foreground/80" : "text-muted-foreground"}>
              {row.slice(0, 6).map((cell, ci) => (
                <td key={ci} className="px-0.5 py-px truncate max-w-[60px] border-b border-muted/30">
                  {cell.slice(0, 20)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="absolute inset-x-0 bottom-0 h-8 bg-gradient-to-t from-card to-transparent" />
    </div>
  );
}

const pdfThumbnailCache = new Map<string, string>();

function usePdfThumbnail(fileId: string) {
  const [dataUrl, setDataUrl] = useState<string | null>(
    pdfThumbnailCache.get(fileId) ?? null,
  );
  const [isLoading, setIsLoading] = useState(!pdfThumbnailCache.has(fileId));
  const [error, setError] = useState(false);

  useEffect(() => {
    if (pdfThumbnailCache.has(fileId)) return;
    let cancelled = false;

    (async () => {
      try {
        const response = await backendApiClient.stream(filesApi.getContentUrl(fileId));
        const data = await response.arrayBuffer();
        const pdf = await pdfjs.getDocument({ data }).promise;
        const page = await pdf.getPage(1);
        const targetWidth = 150;
        const unscaledViewport = page.getViewport({ scale: 1 });
        const scale = targetWidth / unscaledViewport.width;
        const viewport = page.getViewport({ scale });

        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d")!;
        await page.render({ canvasContext: ctx, viewport }).promise;

        const url = canvas.toDataURL("image/jpeg", 0.7);
        pdfThumbnailCache.set(fileId, url);
        if (!cancelled) {
          setDataUrl(url);
          setIsLoading(false);
        }
        pdf.destroy();
      } catch {
        if (!cancelled) {
          setError(true);
          setIsLoading(false);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [fileId]);

  return { dataUrl, isLoading, error };
}

function PdfThumbnail({ fileId, mimeType }: { fileId: string; mimeType: string }) {
  const { dataUrl, isLoading, error } = usePdfThumbnail(fileId);

  if (dataUrl) {
    return <img src={dataUrl} alt="PDF preview" className="w-full h-full object-cover" />;
  }
  if (isLoading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-muted/30">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground/50" />
      </div>
    );
  }
  return (
    <div className="w-full h-full flex flex-col items-center justify-center bg-muted/30 gap-1">
      <FileTypeIcon mimeType={mimeType} className="w-10 h-10" />
      <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wide">PDF</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// FileThumbnail — routes to the right preview
// ---------------------------------------------------------------------------

function FileThumbnail({ file }: { file: UserFile }) {
  const [imgError, setImgError] = useState(false);
  const category = getPreviewCategory(file.mime_type);
  const isImage = category === "image";
  const imageUrl = useImageUrl(file.file_id, isImage);
  const { data: previewText, isLoading } = useTextPreview(file.file_id, file.mime_type);

  if (isImage && !imgError) {
    if (!imageUrl) {
      return (
        <div className="w-full h-full flex items-center justify-center bg-muted/30">
          <Loader2 className="w-5 h-5 animate-spin text-muted-foreground/50" />
        </div>
      );
    }
    return (
      <img
        src={imageUrl}
        alt={file.filename}
        className="w-full h-full object-cover"
        onError={() => setImgError(true)}
        loading="lazy"
      />
    );
  }

  if (category === "pdf") {
    return <PdfThumbnail fileId={file.file_id} mimeType={file.mime_type} />;
  }

  if ((category === "text" || category === "data") && previewText) {
    const isCSV = file.mime_type === "text/csv" || file.filename.toLowerCase().endsWith(".csv");
    const isJSON = file.mime_type === "application/json" || file.filename.toLowerCase().endsWith(".json");

    if (isCSV) return <CsvThumbnail text={previewText} />;
    if (isJSON) return <JsonThumbnail text={previewText} />;
    return <TextThumbnail text={previewText} />;
  }

  if (isLoading && shouldFetchPreview(file.mime_type)) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-muted/30">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground/50" />
      </div>
    );
  }

  return (
    <div className="w-full h-full flex items-center justify-center bg-muted/30">
      <FileTypeIcon mimeType={file.mime_type} className="w-10 h-10" />
    </div>
  );
}

export function FileCard(props: FileItemProps) {
  const { file, isSelected, onToggleSelect, onPreview, onDownload, onCopyLink, onDelete, isDeleting } = props;

  return (
    <div
      className={cn(
        "group relative rounded-lg border bg-card overflow-hidden transition-all hover:shadow-md",
        isSelected && "ring-2 ring-primary"
      )}
    >
      <div className="absolute top-2 left-2 z-10">
        <Checkbox checked={isSelected} onCheckedChange={onToggleSelect} className="bg-background/80 backdrop-blur-sm" />
      </div>
      <div role="button" tabIndex={0} className="w-full aspect-[4/3] overflow-hidden cursor-pointer" onClick={onPreview} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPreview(); } }}>
        <FileThumbnail file={file} />
        <div className="absolute inset-0 top-0 bg-black/0 group-hover:bg-black/40 transition-colors flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100">
          <Button variant="secondary" size="sm" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); onPreview(); }} title="Preview">
            <Eye className="h-4 w-4" />
          </Button>
          <Button variant="secondary" size="sm" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); onDownload(); }} title="Download">
            <Download className="h-4 w-4" />
          </Button>
          <Button variant="secondary" size="sm" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); onCopyLink(); }} title="Copy link">
            <Link2 className="h-4 w-4" />
          </Button>
          <Button variant="secondary" size="sm" className="h-8 w-8 p-0" onClick={(e) => { e.stopPropagation(); onDelete(); }} disabled={isDeleting} title="Delete">
            {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>
      <div className="p-2.5 space-y-1">
        <p className="text-sm font-medium truncate" title={file.filename}>{file.filename}</p>
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{file.size_human}</span>
          <span>{formatRelativeDate(file.created_at)}</span>
        </div>
      </div>
    </div>
  );
}

export function FileRow(props: FileItemProps) {
  const { file, isSelected, onToggleSelect, onPreview, onDownload, onCopyLink, onDelete, isDeleting } = props;

  return (
    <div className={cn("flex items-center gap-3 py-2.5 px-3 rounded-lg transition-colors", isSelected ? "bg-primary/5" : "hover:bg-muted/50")}>
      <Checkbox checked={isSelected} onCheckedChange={onToggleSelect} />
      <button type="button" className="flex-1 min-w-0 flex items-center gap-3 cursor-pointer text-left" onClick={onPreview}>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{file.filename}</p>
          {file.workflow_name && <p className="text-xs text-muted-foreground truncate">From: {file.workflow_name}</p>}
        </div>
        <FileTypeBadge mimeType={file.mime_type} className="flex-shrink-0" />
        <span className="text-xs text-muted-foreground w-16 text-right flex-shrink-0">{file.size_human}</span>
        <span className="text-xs text-muted-foreground w-20 text-right flex-shrink-0">{formatRelativeDate(file.created_at)}</span>
      </button>
      <div className="flex items-center gap-1 flex-shrink-0">
        <Button variant="ghost" size="sm" onClick={onPreview} className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground" title="Preview">
          <Eye className="h-3.5 w-3.5" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onDownload} className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground" title="Download">
          <Download className="h-3.5 w-3.5" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onCopyLink} className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground" title="Copy link">
          <Link2 className="h-3.5 w-3.5" />
        </Button>
        <Button variant="ghost" size="sm" onClick={onDelete} disabled={isDeleting} className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive" title="Delete">
          {isDeleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
        </Button>
      </div>
    </div>
  );
}

export function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  return (
    <div className="py-16 text-center">
      <div className="w-12 h-12 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-3">
        <FolderOpen className="h-6 w-6 text-muted-foreground/50" />
      </div>
      <p className="text-sm text-muted-foreground">{hasFilters ? "No files match your filters" : "No files yet"}</p>
      <p className="text-xs text-muted-foreground mt-1">{hasFilters ? "Try adjusting your search or filters" : "Upload files to get started"}</p>
    </div>
  );
}

export function GridSkeleton() {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
      {Array.from({ length: 12 }).map((_, i) => (
        <div key={i} className="rounded-lg border bg-card overflow-hidden">
          <div className="aspect-[4/3] bg-muted animate-pulse" />
          <div className="p-2.5 space-y-1.5">
            <div className="h-4 bg-muted rounded w-3/4 animate-pulse" />
            <div className="h-3 bg-muted rounded w-1/2 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function ListSkeleton() {
  return (
    <div className="space-y-2">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 py-2.5 px-3">
          <div className="w-4 h-4 bg-muted rounded animate-pulse" />
          <div className="flex-1 space-y-1">
            <div className="h-4 bg-muted rounded w-1/3 animate-pulse" />
            <div className="h-3 bg-muted rounded w-1/4 animate-pulse" />
          </div>
          <div className="h-5 w-12 bg-muted rounded animate-pulse" />
        </div>
      ))}
    </div>
  );
}
