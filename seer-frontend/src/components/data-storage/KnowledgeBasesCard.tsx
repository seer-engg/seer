import { useState } from "react";
import { Database, Plus, ChevronRight } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { knowledgeApi, type KnowledgeBase } from "@/lib/knowledge-api";
import { knowledgeKeys } from "@/lib/query-keys";
import { CreateKnowledgeBaseDialog } from "./CreateKnowledgeBaseDialog";
import { KnowledgeBaseDetailDialog } from "./KnowledgeBaseDetailDialog";

function KnowledgeBaseRow({ kb, onClick }: { kb: KnowledgeBase; onClick: () => void }) {
  return (
    <button onClick={onClick} className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors text-left">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-seer/10 flex items-center justify-center">
          <Database className="h-4 w-4 text-seer" />
        </div>
        <div>
          <p className="text-sm font-medium">{kb.name}</p>
          {kb.description && <p className="text-xs text-muted-foreground line-clamp-1">{kb.description}</p>}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
          {kb.document_count} {kb.document_count === 1 ? "doc" : "docs"}
        </span>
        <ChevronRight className="h-4 w-4 text-muted-foreground" />
      </div>
    </button>
  );
}

function CardHeaderContent({ isLoading }: { isLoading?: boolean }) {
  return (
    <CardHeader>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
          <Database className="h-5 w-5 text-seer" />
        </div>
        <div className="flex-1">
          <CardTitle className="text-base">Knowledge Bases</CardTitle>
          <CardDescription>{isLoading ? "Loading..." : "Manage your document collections for AI-powered search"}</CardDescription>
        </div>
      </div>
    </CardHeader>
  );
}

export function KnowledgeBasesCard() {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [selectedKb, setSelectedKb] = useState<KnowledgeBase | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);

  const { data, isLoading } = useQuery({
    queryKey: knowledgeKeys.lists(),
    queryFn: knowledgeApi.listKnowledgeBases,
  });
  const knowledgeBases = data?.items ?? [];

  const handleDetailClose = (open: boolean) => {
    setDetailDialogOpen(open);
    if (!open) setTimeout(() => setSelectedKb(null), 200);
  };

  if (isLoading) return <Card><CardHeaderContent isLoading /></Card>;

  return (
    <>
      <Card>
        <CardHeaderContent />
        <CardContent className="space-y-4">
          {knowledgeBases.length === 0 ? (
            <div className="py-6 text-center">
              <Database className="h-10 w-10 text-muted-foreground/40 mx-auto mb-3" />
              <p className="text-sm text-muted-foreground">No knowledge bases yet</p>
              <p className="text-xs text-muted-foreground mt-1">Create one to upload and search your documents</p>
            </div>
          ) : (
            <div className="space-y-1">
              {knowledgeBases.map((kb) => (
                <KnowledgeBaseRow key={kb.kb_id} kb={kb} onClick={() => { setSelectedKb(kb); setDetailDialogOpen(true); }} />
              ))}
            </div>
          )}
          <Button onClick={() => setCreateDialogOpen(true)} className="w-full" variant="outline">
            <Plus className="h-4 w-4 mr-2" />New Knowledge Base
          </Button>
        </CardContent>
      </Card>
      <CreateKnowledgeBaseDialog open={createDialogOpen} onOpenChange={setCreateDialogOpen} />
      <KnowledgeBaseDetailDialog open={detailDialogOpen} onOpenChange={handleDetailClose} knowledgeBase={selectedKb} />
    </>
  );
}
