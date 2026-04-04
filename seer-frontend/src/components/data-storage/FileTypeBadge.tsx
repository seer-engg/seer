import { FileText, Image, FileJson, File } from "lucide-react";
import { cn } from "@/lib/utils";
import { getFileTypeCategory } from "@/lib/files-api";

type FileTypeCategory = "pdf" | "image" | "document" | "data" | "other";

interface FileTypeBadgeProps {
  mimeType: string;
  className?: string;
  showLabel?: boolean;
}

const typeConfig: Record<
  FileTypeCategory,
  {
    label: string;
    icon: typeof FileText;
    className: string;
  }
> = {
  pdf: {
    label: "PDF",
    icon: FileText,
    className: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  },
  image: {
    label: "Image",
    icon: Image,
    className: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
  },
  document: {
    label: "Doc",
    icon: FileText,
    className: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  },
  data: {
    label: "Data",
    icon: FileJson,
    className: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
  },
  other: {
    label: "File",
    icon: File,
    className: "bg-muted text-muted-foreground border-border",
  },
};

// Map common MIME types to display labels
function getMimeTypeLabel(mimeType: string): string {
  const mimeLabels: Record<string, string> = {
    "application/pdf": "PDF",
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/gif": "GIF",
    "image/webp": "WebP",
    "image/svg+xml": "SVG",
    "text/plain": "Text",
    "text/markdown": "Markdown",
    "text/csv": "CSV",
    "application/json": "JSON",
    "application/xml": "XML",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "XLSX",
    "application/msword": "DOC",
    "application/vnd.ms-excel": "XLS",
  };

  return mimeLabels[mimeType] || typeConfig[getFileTypeCategory(mimeType)].label;
}

export function FileTypeBadge({
  mimeType,
  className,
  showLabel = true,
}: FileTypeBadgeProps) {
  const category = getFileTypeCategory(mimeType);
  const config = typeConfig[category];
  const Icon = config.icon;
  const label = getMimeTypeLabel(mimeType);

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium border",
        config.className,
        className
      )}
    >
      <Icon className="h-3 w-3" />
      {showLabel && <span>{label}</span>}
    </div>
  );
}

// Icon-only variant for compact displays
export function FileTypeIcon({
  mimeType,
  className,
}: Omit<FileTypeBadgeProps, "showLabel">) {
  const category = getFileTypeCategory(mimeType);
  const config = typeConfig[category];
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "inline-flex items-center justify-center w-6 h-6 rounded",
        config.className,
        className
      )}
    >
      <Icon className="h-3.5 w-3.5" />
    </div>
  );
}

export type { FileTypeBadgeProps };
