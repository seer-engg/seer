import { useState, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/utility/use-toast";
import { filesApi } from "@/lib/files-api";
import { fileKeys } from "@/lib/query-keys";

export function usePagination() {
  const [cursor, setCursor] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const goNext = useCallback(
    (nextCursor: string) => {
      setHistory((p) => [...p, cursor || ""]);
      setCursor(nextCursor);
    },
    [cursor]
  );
  const goPrev = useCallback(() => {
    const h = [...history];
    const prev = h.pop() || null;
    setHistory(h);
    setCursor(prev === "" ? null : prev);
  }, [history]);
  const reset = useCallback(() => {
    setCursor(null);
    setHistory([]);
  }, []);
  return {
    cursor,
    currentPage: history.length + 1,
    hasPrev: history.length > 0,
    goNext,
    goPrev,
    reset,
  };
}

export function useFileOperations() {
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

  const handleDownload = useCallback(
    async (fileId: string) => {
      try {
        const response = await filesApi.getDownloadUrl(fileId);
        window.open(response.download_url, "_blank");
      } catch (error) {
        toast({
          title: "Download failed",
          description: error instanceof Error ? error.message : "Failed to get download URL",
          variant: "destructive",
        });
      }
    },
    [toast]
  );

  const handleCopyLink = useCallback(
    async (fileId: string) => {
      try {
        const response = await filesApi.getDownloadUrl(fileId, true);
        await navigator.clipboard.writeText(response.download_url);
        toast({ title: "Link copied", description: "Download link copied to clipboard." });
      } catch (error) {
        toast({
          title: "Copy failed",
          description: error instanceof Error ? error.message : "Failed to copy link",
          variant: "destructive",
        });
      }
    },
    [toast]
  );

  const handleDelete = useCallback(
    (fileId: string) => {
      setDeletingFileId(fileId);
      deleteMutation.mutate(fileId);
    },
    [deleteMutation]
  );

  return { deletingFileId, bulkDeleteMutation, handleDownload, handleCopyLink, handleDelete };
}
