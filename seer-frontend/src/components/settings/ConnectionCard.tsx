import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Trash2, Loader2 } from "lucide-react";
import { formatScopeName, getProviderDisplayName } from "@/lib/oauth-helpers";
import type { ConnectedAccount } from "@/lib/api-client";

interface ConnectionCardProps {
  connection: ConnectedAccount;
  isDeleting: boolean;
  isLastCard: boolean;
  onDelete: (conn: ConnectedAccount) => void;
}

export function ConnectionCard({ connection, isDeleting, isLastCard, onDelete }: ConnectionCardProps) {
  const provider = connection.provider || connection.toolkit?.slug || 'unknown';
  const scopes = connection.scopes ? connection.scopes.split(/[\s,]+/).filter(Boolean) : [];
  const isActive = connection.status === 'ACTIVE';

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{getProviderDisplayName(provider)}</span>
          <Badge
            variant={isActive ? "default" : "secondary"}
            className={isActive ? "bg-success/10 text-success border-success/20" : ""}
          >
            {connection.status}
          </Badge>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onDelete(connection)}
          disabled={isDeleting}
          className="h-8 w-8 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
        >
          {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
        </Button>
      </div>

      {connection.id && (
        <div className="text-xs text-muted-foreground">Connection ID: {connection.id}</div>
      )}

      {scopes.length > 0 ? (
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">
            Granted Scopes ({scopes.length}):
          </div>
          <div className="max-h-48 overflow-y-auto space-y-1.5 pl-2 border-l-2 border-secondary">
            {scopes.map((scope, idx) => (
              <div key={idx} className="text-xs text-left">
                <span className="text-foreground/70">• </span>
                <span className="font-semibold text-foreground">{formatScopeName(scope)}</span>
                <span className="text-muted-foreground/60 ml-1">({scope})</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">No scopes granted</div>
      )}

      {!isLastCard && <Separator className="my-4" />}
    </div>
  );
}
