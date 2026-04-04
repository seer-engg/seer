import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCurrentUser } from '@/hooks/useAuthProvider';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { Shield, Trash2, Loader2, Plus, ChevronDown, ChevronRight } from 'lucide-react';
import {
  getConnectionsByProvider,
  deleteConnectedAccount,
  initiateConnection,
  type AccountInfo,
} from '@/lib/api-client';
import { connectionKeys } from '@/lib/query-keys';
import { formatScopeName, getProviderDisplayName } from '@/lib/oauth-helpers';
import { useToast } from '@/hooks/utility/use-toast';
import { cn } from '@/lib/utils';

const DEFAULT_PROVIDER_SCOPES: Record<string, string> = {
  google: 'https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile',
  github: 'read:user user:email',
  slack: 'users:read',
  discord: 'identify email',
};

function getAccountDisplayLabel(account: AccountInfo): string {
  return account.display_name?.trim() || account.provider_account_id;
}

function formatAccountStatus(status: string): string {
  const normalizedStatus = status.trim().toLowerCase().replace(/[_-]+/g, ' ');
  return normalizedStatus.replace(/\b\w/g, (char) => char.toUpperCase());
}

function AccountRow({ account, isDeleting, onDelete }: {
  account: AccountInfo;
  isDeleting: boolean;
  onDelete: (account: AccountInfo) => void;
}) {
  const scopes = account.scopes ? account.scopes.split(/[\s,]+/).filter(Boolean) : [];
  const accountLabel = getAccountDisplayLabel(account);
  const formattedStatus = formatAccountStatus(account.status);
  const isActive = account.status.trim().toLowerCase() === 'active';
  const [showScopes, setShowScopes] = useState(false);

  return (
    <div className="p-3 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-medium truncate">{accountLabel}</span>
          </div>
          {account.display_name && account.provider_account_id !== accountLabel && (
            <div className="text-xs text-muted-foreground truncate">
              Account ID: {account.provider_account_id}
            </div>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge
            variant={isActive ? 'default' : 'secondary'}
            className={cn(
              'h-5 px-1.5 text-[10px] flex-shrink-0',
              isActive && 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20',
            )}
          >
            {formattedStatus}
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onDelete(account)}
            disabled={isDeleting}
            aria-label={`Disconnect ${accountLabel}`}
            className="h-8 w-8 p-0 text-destructive hover:text-destructive hover:bg-destructive/10"
          >
            {isDeleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </Button>
        </div>
      </div>
      {scopes.length > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            className="inline-flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowScopes(!showScopes)}
          >
            <span>{showScopes ? 'Hide' : 'Show'} granted scopes</span>
            <Badge
              variant="secondary"
              className="h-5 rounded-md px-1.5 text-[10px] bg-muted text-muted-foreground border-border/60"
            >
              {scopes.length}
            </Badge>
          </button>
          {showScopes && (
            <div className="rounded-xl border border-border/60 bg-muted/30 p-3">
              <div className="flex flex-wrap gap-1.5">
                {scopes.map((scope, idx) => (
                  <Badge
                    key={`${scope}-${idx}`}
                    variant="secondary"
                    className="h-6 rounded-md px-2 text-[11px] font-medium bg-secondary/70 text-secondary-foreground border-border/60"
                  >
                    {formatScopeName(scope)}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function ProviderGroup({ provider, accounts, deletingId, onDelete, onAddAccount, addingProvider }: {
  provider: string;
  accounts: AccountInfo[];
  deletingId: number | null;
  onDelete: (account: AccountInfo) => void;
  onAddAccount: (provider: string) => void;
  addingProvider: string | null;
}) {
  const [expanded, setExpanded] = useState(true);
  const displayName = getProviderDisplayName(provider);

  return (
    <div className="border rounded-lg overflow-hidden">
      <button type="button" className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors" onClick={() => setExpanded(!expanded)}>
        <div className="flex items-center gap-2">
          {expanded ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
          <span className="font-medium">{displayName}</span>
          <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">{accounts.length} {accounts.length === 1 ? 'account' : 'accounts'}</Badge>
        </div>
      </button>
      {expanded && (
        <div className="border-t">
          <div className="divide-y">
            {accounts.map((account) => (
              <AccountRow key={account.id} account={account} isDeleting={deletingId === account.id} onDelete={onDelete} />
            ))}
          </div>
          <div className="p-3 bg-muted/30">
            <Button variant="ghost" size="sm" className="h-8 text-xs text-seer hover:text-seer hover:bg-seer/10" onClick={() => onAddAccount(provider)} disabled={addingProvider === provider}>
              {addingProvider === provider ? <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" /> : <Plus className="h-3.5 w-3.5 mr-1.5" />}
              Add another {displayName} account
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function DeleteDialog({ open, onOpenChange, account, onConfirm }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  account: AccountInfo | null;
  onConfirm: () => void;
}) {
  const accountLabel = account ? getAccountDisplayLabel(account) : null;

  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete Connection?</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to disconnect <span className="font-medium">{accountLabel}</span>? Workflows using this account will need to be reconfigured.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">Delete</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function ConnectionsList({ providers, connections, deletingId, onDelete, onAddAccount, addingProvider }: {
  providers: string[];
  connections: Record<string, AccountInfo[]>;
  deletingId: number | null;
  onDelete: (account: AccountInfo) => void;
  onAddAccount: (provider: string) => void;
  addingProvider: string | null;
}) {
  return (
    <div className="space-y-3">
      {providers.map((provider) => (
        <ProviderGroup
          key={provider}
          provider={provider}
          accounts={connections[provider]}
          deletingId={deletingId}
          onDelete={onDelete}
          onAddAccount={onAddAccount}
          addingProvider={addingProvider}
        />
      ))}
    </div>
  );
}

function CardContentArea({ isLoading, error, providers, data, deletingId, onDelete, onAddAccount, addingProvider }: {
  isLoading: boolean;
  error: Error | null;
  providers: string[];
  data: { connections: Record<string, AccountInfo[]> } | undefined;
  deletingId: number | null;
  onDelete: (account: AccountInfo) => void;
  onAddAccount: (provider: string) => void;
  addingProvider: string | null;
}) {
  if (isLoading) return <div className="flex items-center gap-2 text-sm text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" />Loading connections...</div>;
  if (error) return <div className="text-sm text-destructive">Failed to load connections. Please try again later.</div>;
  if (providers.length === 0) return <div className="text-sm text-muted-foreground">No connected accounts found. Connect an account from a workflow to get started.</div>;
  return <ConnectionsList providers={providers} connections={data!.connections} deletingId={deletingId} onDelete={onDelete} onAddAccount={onAddAccount} addingProvider={addingProvider} />;
}

export function GroupedConnectionsCard() {
  const { toast } = useToast();
  const { user } = useCurrentUser();
  const queryClient = useQueryClient();

  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [accountToDelete, setAccountToDelete] = useState<AccountInfo | null>(null);
  const [addingProvider, setAddingProvider] = useState<string | null>(null);

  const userEmail = user?.primaryEmailAddress?.emailAddress ?? user?.emailAddresses?.[0]?.emailAddress ?? null;
  const { data, isLoading, error } = useQuery({
    queryKey: connectionKeys.grouped(),
    queryFn: getConnectionsByProvider,
    staleTime: 30000,
  });

  const handleDeleteClick = (account: AccountInfo) => { setAccountToDelete(account); setDeleteDialogOpen(true); };

  const handleDeleteConfirm = async () => {
    if (!accountToDelete) return;
    setDeleteDialogOpen(false);
    setDeletingId(accountToDelete.id);
    try {
      await deleteConnectedAccount(String(accountToDelete.id));
      await queryClient.invalidateQueries({ queryKey: connectionKeys.grouped() });
      toast({
        title: 'Connection deleted',
        description: `Your ${getProviderDisplayName(accountToDelete.provider)} account (${getAccountDisplayLabel(accountToDelete)}) has been disconnected.`,
      });
    } catch (err) {
      toast({ title: 'Failed to delete connection', description: err instanceof Error ? err.message : 'An error occurred.', variant: 'destructive' });
    } finally {
      setDeletingId(null);
      setAccountToDelete(null);
    }
  };

  const handleAddAccount = async (provider: string) => {
    if (!userEmail) { toast({ title: 'Not logged in', description: 'Please sign in to connect an account.', variant: 'destructive' }); return; }
    setAddingProvider(provider);
    try {
      const response = await initiateConnection({
        userId: userEmail,
        provider,
        scope: DEFAULT_PROVIDER_SCOPES[provider] || '',
        callbackUrl: `${window.location.origin}${window.location.pathname}`,
        forceNewAccount: true,  // Always force new account flow when clicking "Add another account"
      });
      if (response.redirectUrl) window.location.href = response.redirectUrl;
    } catch (err) {
      toast({ title: 'Connection failed', description: err instanceof Error ? err.message : 'Failed to start OAuth flow.', variant: 'destructive' });
      setAddingProvider(null);
    }
  };

  const providers = data?.connections ? Object.keys(data.connections).sort() : [];

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center"><Shield className="h-5 w-5 text-seer" /></div>
            <div>
              <CardTitle className="text-base">Connected Accounts</CardTitle>
              <CardDescription>Manage your connected accounts by provider</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <CardContentArea isLoading={isLoading} error={error} providers={providers} data={data} deletingId={deletingId} onDelete={handleDeleteClick} onAddAccount={handleAddAccount} addingProvider={addingProvider} />
        </CardContent>
      </Card>
      <DeleteDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen} account={accountToDelete} onConfirm={handleDeleteConfirm} />
    </>
  );
}
