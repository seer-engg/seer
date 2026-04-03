import { useState } from "react";
import { Search, ChevronDown, Loader2 } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { useToast } from "@/hooks/utility/use-toast";
import { cn } from "@/lib/utils";
import { knowledgeApi, type QueryResponse } from "@/lib/knowledge-api";
import { QueryResultCard } from "./QueryResultCard";

interface AdvancedOptionsProps {
  topK: number;
  minScore: number;
  onTopKChange: (value: number) => void;
  onMinScoreChange: (value: number) => void;
}

function AdvancedOptions({ topK, minScore, onTopKChange, onMinScoreChange }: AdvancedOptionsProps) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger asChild>
        <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground">
          <ChevronDown className={cn("h-3 w-3 mr-1 transition-transform", open && "rotate-180")} />
          Advanced options
        </Button>
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-3 space-y-4">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Results count</Label>
            <span className="text-sm text-muted-foreground">{topK}</span>
          </div>
          <Slider value={[topK]} onValueChange={([v]) => onTopKChange(v)} min={1} max={50} step={1} />
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Minimum score</Label>
            <span className="text-sm text-muted-foreground">{minScore.toFixed(1)}</span>
          </div>
          <Slider value={[minScore]} onValueChange={([v]) => onMinScoreChange(v)} min={0} max={1} step={0.1} />
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}

function QueryResults({ results, isPending }: { results: QueryResponse | null; isPending: boolean }) {
  if (isPending) {
    return (
      <div className="py-8 text-center">
        <Loader2 className="h-6 w-6 animate-spin mx-auto text-muted-foreground" />
        <p className="text-sm text-muted-foreground mt-2">Searching...</p>
      </div>
    );
  }

  if (!results) {
    return <div className="py-8 text-center text-sm text-muted-foreground">Enter a query to search this knowledge base.</div>;
  }

  if (results.results.length === 0) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-muted-foreground">No results found</p>
        <div className="py-6 text-center text-sm text-muted-foreground">
          Try a different query or adjust the minimum score threshold.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-sm text-muted-foreground">
        Found {results.results.length} result{results.results.length === 1 ? "" : "s"}
      </p>
      <div className="space-y-3">
        {results.results.map((result) => (
          <QueryResultCard key={result.chunk_id} result={result} />
        ))}
      </div>
    </div>
  );
}

interface KnowledgeBaseQueryTabProps {
  kbId: string;
}

export function KnowledgeBaseQueryTab({ kbId }: KnowledgeBaseQueryTabProps) {
  const { toast } = useToast();
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0.7);
  const [results, setResults] = useState<QueryResponse | null>(null);

  const searchMutation = useMutation({
    mutationFn: () => knowledgeApi.queryKnowledgeBase(kbId, { query, top_k: topK, min_score: minScore }),
    onSuccess: setResults,
    onError: (error: Error) => toast({ title: "Search failed", description: error.message, variant: "destructive" }),
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) searchMutation.mutate();
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <Input placeholder="Ask a question..." value={query} onChange={(e) => setQuery(e.target.value)} className="flex-1" />
          <Button type="submit" disabled={!query.trim() || searchMutation.isPending}>
            {searchMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
          </Button>
        </div>
        <AdvancedOptions topK={topK} minScore={minScore} onTopKChange={setTopK} onMinScoreChange={setMinScore} />
      </form>
      <QueryResults results={results} isPending={searchMutation.isPending} />
    </div>
  );
}

export type { KnowledgeBaseQueryTabProps };
