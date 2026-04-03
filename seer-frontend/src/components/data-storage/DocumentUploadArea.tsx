import { useState, useRef } from "react";
import { Upload, Loader2 } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/utility/use-toast";
import { knowledgeApi, ACCEPTED_DOCUMENT_TYPES, isAcceptedFileType } from "@/lib/knowledge-api";
import { knowledgeKeys } from "@/lib/query-keys";
import { cn } from "@/lib/utils";

interface DocumentUploadAreaProps {
  kbId: string;
  className?: string;
}

export function DocumentUploadArea({ kbId, className }: DocumentUploadAreaProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const uploadMutation = useMutation({
    mutationFn: (file: File) => knowledgeApi.uploadDocument(kbId, file),
    onSuccess: (document) => {
      queryClient.invalidateQueries({ queryKey: knowledgeKeys.documentsByKnowledgeBase(kbId) });
      toast({
        title: "Document uploaded",
        description: `"${document.name}" is being processed.`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Upload failed",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleFileSelect = (file: File | undefined) => {
    if (!file) return;

    if (!isAcceptedFileType(file)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF, TXT, MD, or DOCX file.",
        variant: "destructive",
      });
      return;
    }

    uploadMutation.mutate(file);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    handleFileSelect(file);
    // Reset input so the same file can be selected again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDrag = (e: React.DragEvent, dragging: boolean) => {
    e.preventDefault();
    setIsDragging(dragging);
    if (!dragging && e.type === "drop") handleFileSelect(e.dataTransfer.files?.[0]);
  };

  return (
    <div className={className}>
      <label
        htmlFor={`file-upload-${kbId}`}
        className={cn(
          "flex items-center justify-center w-full px-4 py-6 border-2 border-dashed rounded-lg cursor-pointer transition-colors",
          isDragging
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-primary/50",
          uploadMutation.isPending && "pointer-events-none opacity-50"
        )}
        onDragOver={(e) => handleDrag(e, true)}
        onDragLeave={(e) => handleDrag(e, false)}
        onDrop={(e) => handleDrag(e, false)}
      >
        {uploadMutation.isPending ? (
          <div className="flex flex-col items-center gap-2">
            <Loader2 className="w-8 h-8 text-primary animate-spin" />
            <span className="text-sm text-muted-foreground">Uploading...</span>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="w-8 h-8 text-muted-foreground" />
            <div className="text-center">
              <span className="text-sm font-medium text-foreground">
                Click to upload or drag and drop
              </span>
              <p className="text-xs text-muted-foreground mt-1">
                PDF, TXT, MD, or DOCX
              </p>
            </div>
          </div>
        )}
      </label>
      <input
        ref={fileInputRef}
        id={`file-upload-${kbId}`}
        type="file"
        accept={ACCEPTED_DOCUMENT_TYPES}
        onChange={handleInputChange}
        className="hidden"
        disabled={uploadMutation.isPending}
      />
    </div>
  );
}

export type { DocumentUploadAreaProps };
