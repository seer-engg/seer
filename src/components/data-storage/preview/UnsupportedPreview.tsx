import { FileQuestion, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FileTypeBadge } from "../FileTypeBadge";
import type { UserFile } from "@/lib/files-api";

export interface UnsupportedPreviewProps {
  file: UserFile;
  onDownload: () => void;
}

export function UnsupportedPreview({ file, onDownload }: UnsupportedPreviewProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
        <FileQuestion className="h-8 w-8 text-muted-foreground" />
      </div>
      <h3 className="text-lg font-medium mb-1">{file.filename}</h3>
      <div className="flex items-center gap-2 mb-3">
        <FileTypeBadge mimeType={file.mime_type} />
        <span className="text-sm text-muted-foreground">{file.size_human}</span>
      </div>
      <p className="text-sm text-muted-foreground mb-6">
        Preview not available for this file type
      </p>
      <Button onClick={onDownload} variant="outline">
        <Download className="h-4 w-4 mr-2" />
        Download to view
      </Button>
    </div>
  );
}
