import { useState, useRef } from "react";
import { Upload, Loader2, X, CheckCircle2, AlertCircle } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/utility/use-toast";
import { filesApi, type FileUploadResponse } from "@/lib/files-api";
import { fileKeys } from "@/lib/query-keys";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface FileUploadAreaProps {
  className?: string;
  onUploadComplete?: (file: FileUploadResponse) => void;
}

interface UploadingFile {
  id: string;
  file: File;
  status: "uploading" | "success" | "error";
  error?: string;
}

function UploadItem({ item, onRemove }: { item: UploadingFile; onRemove: () => void }) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 p-2 rounded-md text-sm",
        item.status === "uploading" && "bg-muted/50",
        item.status === "success" && "bg-emerald-500/10",
        item.status === "error" && "bg-destructive/10"
      )}
    >
      {item.status === "uploading" && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
      {item.status === "success" && <CheckCircle2 className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />}
      {item.status === "error" && <AlertCircle className="h-4 w-4 text-destructive" />}
      <span className="flex-1 truncate text-xs">{item.file.name}</span>
      {item.status !== "uploading" && (
        <button type="button" onClick={onRemove} className="p-0.5 hover:bg-muted rounded">
          <X className="h-3 w-3 text-muted-foreground" />
        </button>
      )}
    </div>
  );
}

export function FileUploadArea({ className, onUploadComplete }: FileUploadAreaProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => filesApi.uploadFile(file),
    onSuccess: (result, file) => {
      setUploadingFiles((prev) => prev.map((f) => (f.file === file ? { ...f, status: "success" as const } : f)));
      queryClient.invalidateQueries({ queryKey: fileKeys.all });
      onUploadComplete?.(result);
    },
    onError: (error: Error, file) => {
      setUploadingFiles((prev) => prev.map((f) => (f.file === file ? { ...f, status: "error" as const, error: error.message } : f)));
      toast({ title: "Upload failed", description: `Failed to upload ${file.name}: ${error.message}`, variant: "destructive" });
    },
  });

  const handleFileSelect = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const newFiles: UploadingFile[] = Array.from(files).map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      status: "uploading" as const,
    }));
    setUploadingFiles((prev) => [...prev, ...newFiles]);
    newFiles.forEach((uploadingFile) => uploadMutation.mutate(uploadingFile.file));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFileSelect(e.target.files);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  const removeFile = (fileId: string) => setUploadingFiles((prev) => prev.filter((f) => f.id !== fileId));
  const clearCompleted = () => setUploadingFiles((prev) => prev.filter((f) => f.status === "uploading"));
  const hasCompletedFiles = uploadingFiles.some((f) => f.status === "success" || f.status === "error");
  const uploadingCount = uploadingFiles.filter((f) => f.status === "uploading").length;
  const successCount = uploadingFiles.filter((f) => f.status === "success").length;

  return (
    <div className={cn("space-y-3", className)}>
      <label
        htmlFor="file-upload-area"
        className={cn(
          "flex items-center justify-center w-full px-4 py-6 border-2 border-dashed rounded-lg cursor-pointer transition-colors",
          isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
        )}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center gap-2">
          <Upload className="w-8 h-8 text-muted-foreground" />
          <div className="text-center">
            <span className="text-sm font-medium text-foreground">Click to upload or drag and drop</span>
            <p className="text-xs text-muted-foreground mt-1">Any file type supported</p>
          </div>
        </div>
      </label>
      <input ref={fileInputRef} id="file-upload-area" type="file" multiple onChange={handleInputChange} className="hidden" />

      {uploadingFiles.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">
              {uploadingCount > 0 ? `Uploading ${uploadingCount} file(s)...` : `${successCount} file(s) uploaded`}
            </span>
            {hasCompletedFiles && (
              <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={clearCompleted}>Clear</Button>
            )}
          </div>
          <div className="space-y-1.5 max-h-32 overflow-y-auto">
            {uploadingFiles.map((item) => (
              <UploadItem key={item.id} item={item} onRemove={() => removeFile(item.id)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export type { FileUploadAreaProps };
