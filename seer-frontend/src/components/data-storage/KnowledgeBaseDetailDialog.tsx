import { useState } from "react";
import { FileText, File, Trash2 } from "lucide-react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/utility/use-toast";
import { knowledgeApi, formatFileSize, type KnowledgeBase, type KnowledgeDocument } from "@/lib/knowledge-api";
import { knowledgeKeys } from "@/lib/query-keys";
import { DocumentUploadArea } from "./DocumentUploadArea";
import { ProcessingStatusBadge } from "./ProcessingStatusBadge";
import { KnowledgeBaseQueryTab } from "./KnowledgeBaseQueryTab";

interface KnowledgeBaseDetailDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  knowledgeBase: KnowledgeBase | null;
}

const getFileIcon = (mimeType: string) =>
  mimeType === "application/pdf" || mimeType.includes("word") ? FileText : File;

function DocumentRow({ document, kbId }: { document: KnowledgeDocument; kbId: string }) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showConfirm, setShowConfirm] = useState(false);
  const Icon = getFileIcon(document.mime_type);

  const deleteMutation = useMutation({
    mutationFn: () => knowledgeApi.deleteDocument(kbId, document.doc_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: knowledgeKeys.documentsByKnowledgeBase(kbId) });
      queryClient.invalidateQueries({ queryKey: knowledgeKeys.lists() });
      toast({ title: "Document deleted", description: `"${document.name}" has been removed.` });
    },
    onError: (error: Error) => {
      toast({ title: "Failed to delete document", description: error.message, variant: "destructive" });
    },
  });

  return (
    <>
      <div className="flex items-center justify-between py-3 px-2 rounded-lg hover:bg-muted/50 transition-colors">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <Icon className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium truncate">{document.name}</p>
            <p className="text-xs text-muted-foreground">
              {formatFileSize(document.file_size)}
              {document.chunk_count > 0 && ` • ${document.chunk_count} chunks`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <ProcessingStatusBadge status={document.processing_status} error={document.processing_error} />
          <Button variant="ghost" size="sm" onClick={() => setShowConfirm(true)}
            disabled={deleteMutation.isPending} className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
      <AlertDialog open={showConfirm} onOpenChange={setShowConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{document.name}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => deleteMutation.mutate()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

function DocumentsList({ kbId, documents, isLoading }: { kbId: string; documents: KnowledgeDocument[]; isLoading: boolean }) {
  return (
    <div className="space-y-1">
      <h4 className="text-sm font-medium text-muted-foreground mb-2">Documents</h4>
      {isLoading ? (
        <div className="py-8 text-center text-sm text-muted-foreground">Loading documents...</div>
      ) : documents.length === 0 ? (
        <div className="py-8 text-center text-sm text-muted-foreground">No documents yet. Upload your first document above.</div>
      ) : (
        <div className="space-y-1">{documents.map((doc) => <DocumentRow key={doc.doc_id} document={doc} kbId={kbId} />)}</div>
      )}
    </div>
  );
}

function DangerZone({ onDelete, isDeleting }: { onDelete: () => void; isDeleting: boolean }) {
  return (
    <div className="pt-4 border-t">
      <h4 className="text-sm font-medium text-destructive mb-2">Danger Zone</h4>
      <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-4 flex items-center justify-between">
        <div>
          <p className="text-sm font-medium">Delete Knowledge Base</p>
          <p className="text-xs text-muted-foreground">Permanently delete this knowledge base and all its documents.</p>
        </div>
        <Button variant="destructive" size="sm" onClick={onDelete} disabled={isDeleting}>Delete</Button>
      </div>
    </div>
  );
}

export function KnowledgeBaseDetailDialog({ open, onOpenChange, knowledgeBase }: KnowledgeBaseDetailDialogProps) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [activeTab, setActiveTab] = useState<"documents" | "query">("documents");
  const kbId = knowledgeBase?.kb_id ?? "";

  const { data, isLoading } = useQuery({
    queryKey: knowledgeKeys.documentsByKnowledgeBase(kbId),
    queryFn: () => knowledgeApi.listDocuments(kbId),
    enabled: open && !!kbId,
    refetchInterval: (query) => {
      const docs = query.state.data?.items;
      return docs?.some((d) => d.processing_status === "processing" || d.processing_status === "pending") ? 5000 : false;
    },
  });
  const documents = data?.items ?? [];

  const deleteMutation = useMutation({
    mutationFn: () => knowledgeApi.deleteKnowledgeBase(kbId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: knowledgeKeys.lists() });
      toast({ title: "Knowledge base deleted", description: `"${knowledgeBase?.name}" has been removed.` });
      onOpenChange(false);
    },
    onError: (error: Error) => {
      toast({ title: "Failed to delete knowledge base", description: error.message, variant: "destructive" });
    },
  });

  if (!knowledgeBase) return null;

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-lg max-h-[85vh] flex flex-col">
          <DialogHeader>
            <div className="flex items-start justify-between">
              <div>
                <DialogTitle>{knowledgeBase.name}</DialogTitle>
                {knowledgeBase.description && <DialogDescription className="mt-1">{knowledgeBase.description}</DialogDescription>}
              </div>
              <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                {documents.length} {documents.length === 1 ? "document" : "documents"}
              </span>
            </div>
          </DialogHeader>
          <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)} className="flex-1 flex flex-col overflow-hidden">
            <TabsList className="w-full">
              <TabsTrigger value="documents" className="flex-1">Documents</TabsTrigger>
              <TabsTrigger value="query" className="flex-1">Test Query</TabsTrigger>
            </TabsList>
            <TabsContent value="documents" className="mt-4 flex-1 overflow-y-auto space-y-4">
              <DocumentUploadArea kbId={kbId} />
              <DocumentsList kbId={kbId} documents={documents} isLoading={isLoading} />
              <DangerZone onDelete={() => setShowDeleteConfirm(true)} isDeleting={deleteMutation.isPending} />
            </TabsContent>
            <TabsContent value="query" className="mt-4 flex-1 overflow-y-auto">
              <KnowledgeBaseQueryTab kbId={kbId} />
            </TabsContent>
          </Tabs>
        </DialogContent>
      </Dialog>
      <AlertDialog open={showDeleteConfirm} onOpenChange={setShowDeleteConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Knowledge Base?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{knowledgeBase.name}"? This will permanently delete all {documents.length} document(s).
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={() => deleteMutation.mutate()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete Knowledge Base</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}

export type { KnowledgeBaseDetailDialogProps };
