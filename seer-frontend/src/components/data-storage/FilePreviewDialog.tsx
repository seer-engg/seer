import { useState, useEffect, useCallback, useMemo } from "react";
import { Download, X, Loader2, RefreshCw } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/utility/use-toast";
import {
  filesApi,
  getPreviewCategory,
  type UserFile,
  type PreviewCategory,
} from "@/lib/files-api";
import { FileTypeBadge } from "./FileTypeBadge";
import {
  ImagePreview,
  TextPreview,
  DataPreview,
  PdfPreview,
  UnsupportedPreview,
} from "./preview";

export interface FilePreviewDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  file: UserFile | null;
}

interface PreviewState {
  url: string | null;
  isLoading: boolean;
  error: string | null;
}

/**
 * Determines if a file type should use the backend content proxy.
 * Text and data files use the proxy to avoid CORS issues with S3.
 * Images and PDFs can use presigned URLs directly (img/object tags don't have CORS).
 */
function shouldUseContentProxy(category: PreviewCategory): boolean {
  return category === "text" || category === "data";
}

function PreviewContent({
  file,
  url,
  category,
  onDownload,
}: {
  file: UserFile;
  url: string;
  category: PreviewCategory;
  onDownload: () => void;
}) {
  switch (category) {
    case "image":
      return <ImagePreview url={url} filename={file.filename} />;
    case "text":
      return (
        <TextPreview url={url} filename={file.filename} mimeType={file.mime_type} />
      );
    case "data":
      return (
        <DataPreview url={url} filename={file.filename} mimeType={file.mime_type} />
      );
    case "pdf":
      return <PdfPreview url={url} filename={file.filename} />;
    default:
      return <UnsupportedPreview file={file} onDownload={onDownload} />;
  }
}

function PreviewDialogHeader({
  file,
  onDownload,
  onClose,
}: {
  file: UserFile;
  onDownload: () => void;
  onClose: () => void;
}) {
  return (
    <DialogHeader className="px-6 py-4 border-b flex-shrink-0">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <DialogTitle className="truncate text-base">
            {file.filename}
          </DialogTitle>
          <FileTypeBadge mimeType={file.mime_type} />
          <span className="text-sm text-muted-foreground flex-shrink-0">
            {file.size_human}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={onDownload}
            className="h-8"
          >
            <Download className="h-4 w-4 mr-1.5" />
            Download
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-8 w-8 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </DialogHeader>
  );
}

function PreviewErrorState({
  error,
  onRetry,
  onDownload,
}: {
  error: string;
  onRetry: () => void;
  onDownload: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <p className="text-sm text-destructive mb-2">Failed to load preview</p>
      <p className="text-xs text-muted-foreground mb-4">{error}</p>
      <div className="flex gap-2">
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCw className="h-4 w-4 mr-1.5" />
          Retry
        </Button>
        <Button variant="outline" size="sm" onClick={onDownload}>
          <Download className="h-4 w-4 mr-1.5" />
          Download instead
        </Button>
      </div>
    </div>
  );
}

export function FilePreviewDialog({
  open,
  onOpenChange,
  file,
}: FilePreviewDialogProps) {
  const { toast } = useToast();
  const [previewState, setPreviewState] = useState<PreviewState>({
    url: null,
    isLoading: false,
    error: null,
  });

  const category = useMemo(
    () => (file ? getPreviewCategory(file.mime_type) : "other"),
    [file]
  );

  const fetchPreviewUrl = useCallback(async () => {
    if (!file) return;

    setPreviewState({ url: null, isLoading: true, error: null });

    try {
      if (shouldUseContentProxy(getPreviewCategory(file.mime_type))) {
        const contentUrl = filesApi.getContentUrl(file.file_id);
        setPreviewState({ url: contentUrl, isLoading: false, error: null });
      } else {
        const response = await filesApi.getDownloadUrl(file.file_id, true);
        setPreviewState({ url: response.download_url, isLoading: false, error: null });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to load preview";
      setPreviewState({ url: null, isLoading: false, error: message });
    }
  }, [file]);

  useEffect(() => {
    if (open && file) {
      fetchPreviewUrl();
    } else {
      setPreviewState({ url: null, isLoading: false, error: null });
    }
  }, [open, file, fetchPreviewUrl]);

  const handleDownload = useCallback(async () => {
    if (!file) return;
    try {
      const response = await filesApi.getDownloadUrl(file.file_id, false);
      window.open(response.download_url, "_blank");
    } catch (err) {
      toast({
        title: "Download failed",
        description: err instanceof Error ? err.message : "Failed to get download URL",
        variant: "destructive",
      });
    }
  }, [file, toast]);

  const handleClose = useCallback(() => onOpenChange(false), [onOpenChange]);

  if (!file) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-4xl max-h-[90vh] flex flex-col p-0">
        <PreviewDialogHeader file={file} onDownload={handleDownload} onClose={handleClose} />
        <div className="flex-1 min-h-0 overflow-hidden">
          {previewState.isLoading && (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          )}
          {previewState.error && (
            <PreviewErrorState
              error={previewState.error}
              onRetry={fetchPreviewUrl}
              onDownload={handleDownload}
            />
          )}
          {previewState.url && !previewState.isLoading && !previewState.error && (
            <PreviewContent
              file={file}
              url={previewState.url}
              category={category}
              onDownload={handleDownload}
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
