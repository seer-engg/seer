import { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { QueryResult } from "@/lib/knowledge-api";

interface QueryResultCardProps {
  result: QueryResult;
}

function getScoreColor(score: number): string {
  if (score >= 0.8) {
    return "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20";
  }
  if (score >= 0.6) {
    return "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20";
  }
  return "bg-muted text-muted-foreground border-muted";
}

const CONTENT_PREVIEW_LENGTH = 200;

export function QueryResultCard({ result }: QueryResultCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const needsTruncation = result.content.length > CONTENT_PREVIEW_LENGTH;

  const displayContent = isExpanded || !needsTruncation
    ? result.content
    : result.content.slice(0, CONTENT_PREVIEW_LENGTH) + "...";

  return (
    <div className="rounded-lg border bg-card p-4 space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
          <span className="text-sm font-medium truncate">{result.doc_name}</span>
        </div>
        <Badge variant="outline" className={cn("flex-shrink-0", getScoreColor(result.score))}>
          {(result.score * 100).toFixed(0)}% match
        </Badge>
      </div>

      <p className="text-sm text-muted-foreground whitespace-pre-wrap break-words">
        {displayContent}
      </p>

      {needsTruncation && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show more
            </>
          )}
        </Button>
      )}
    </div>
  );
}

export type { QueryResultCardProps };
