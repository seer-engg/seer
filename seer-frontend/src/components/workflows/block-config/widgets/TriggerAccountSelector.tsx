import { useState, useEffect, useCallback } from 'react';
import { Check, ChevronsUpDown, AlertTriangle, Link2 } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { getTriggerAccounts, type ToolAccountInfo, type ToolAccountsResponse } from '@/lib/api-client';

export interface TriggerAccountSelectorProps {
  triggerKey: string;
  selectedConnectionId?: number | null;
  onSelect?: (connectionId: number | null) => void;
  disabled?: boolean;
  error?: string;
  onConnectAccount?: () => void;
}

function AccountDisplay({ account, showScopeWarning = false }: { account: ToolAccountInfo; showScopeWarning?: boolean }) {
  return (
    <div className="flex items-center gap-2 min-w-0">
      <span className="truncate">{account.display_name}</span>
      {showScopeWarning && !account.has_required_scopes && (
        <Tooltip>
          <TooltipTrigger asChild>
            <AlertTriangle className="h-3.5 w-3.5 text-amber-500 flex-shrink-0" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs text-xs">
            <p className="font-medium mb-1">Missing scopes:</p>
            <ul className="list-disc list-inside">
              {account.missing_scopes.map((scope) => (<li key={scope}>{scope}</li>))}
            </ul>
          </TooltipContent>
        </Tooltip>
      )}
      {showScopeWarning && account.has_required_scopes && (
        <Check className="h-3.5 w-3.5 text-emerald-500 flex-shrink-0" />
      )}
    </div>
  );
}

function AccountLabel({ children }: { children: React.ReactNode }) {
  return <Label className="text-xs font-medium">{children}</Label>;
}

function useTriggerAccounts(triggerKey: string, selectedConnectionId: number | null | undefined, onSelect?: (id: number | null) => void) {
  const [loading, setLoading] = useState(false);
  const [accountsData, setAccountsData] = useState<ToolAccountsResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    if (!triggerKey) { setAccountsData(null); return; }
    let cancelled = false;
    setLoading(true);
    setFetchError(null);
    getTriggerAccounts(triggerKey)
      .then((response) => { if (!cancelled) setAccountsData(response); })
      .catch((err) => { if (!cancelled) setFetchError(err instanceof Error ? err.message : 'Failed to fetch accounts'); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [triggerKey]);

  // Auto-select first account when nothing selected
  useEffect(() => {
    if (accountsData && accountsData.accounts.length >= 1 && selectedConnectionId == null && onSelect) {
      onSelect(accountsData.accounts[0].id);
    }
  }, [accountsData, selectedConnectionId, onSelect]);

  return { loading, accountsData, fetchError };
}

/** Account selector dropdown for multi-account OAuth support on triggers. */
export function TriggerAccountSelector({ triggerKey, selectedConnectionId, onSelect, disabled = false, error, onConnectAccount }: TriggerAccountSelectorProps) {
  const [open, setOpen] = useState(false);
  const { loading, accountsData, fetchError } = useTriggerAccounts(triggerKey, selectedConnectionId, onSelect);

  const handleSelect = useCallback((connectionId: number) => { onSelect?.(connectionId); setOpen(false); }, [onSelect]);

  if (loading && !accountsData) return (
    <div className="space-y-1.5">
      <AccountLabel>Account</AccountLabel>
      <div className="text-xs text-muted-foreground animate-pulse">Loading accounts...</div>
    </div>
  );
  if (fetchError) return (
    <div className="space-y-1.5">
      <AccountLabel>Account</AccountLabel>
      <div className="text-xs text-destructive">{fetchError}</div>
    </div>
  );
  if (accountsData && accountsData.accounts.length === 0) return (
    <div className="space-y-1.5">
      <AccountLabel>Account</AccountLabel>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
        <span>No connected account</span>
        {onConnectAccount && (
          <Button variant="link" size="sm" className="h-auto p-0 text-xs text-seer" onClick={onConnectAccount}>
            <Link2 className="h-3 w-3 mr-1" />Connect Account
          </Button>
        )}
      </div>
    </div>
  );
  if (!accountsData) return null;

  const selectedAccount = accountsData.accounts.find((acc) => acc.id === selectedConnectionId);

  return (
    <div className="space-y-1.5">
      <AccountLabel>Account{accountsData.requires_selection && <span className="text-destructive ml-1">*</span>}</AccountLabel>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button variant="outline" role="combobox" aria-expanded={open} disabled={disabled}
            className={cn('w-full justify-between font-normal text-left', !selectedAccount && 'text-muted-foreground', error && 'border-destructive focus-visible:ring-destructive')}>
            {selectedAccount ? <AccountDisplay account={selectedAccount} /> : 'Select an account...'}
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search accounts..." />
            <CommandList>
              <CommandEmpty>No account found.</CommandEmpty>
              <CommandGroup>
                {accountsData.accounts.map((account) => (
                  <CommandItem key={account.id} value={account.display_name} onSelect={() => handleSelect(account.id)} className="flex items-center gap-2">
                    <Check className={cn('h-4 w-4 flex-shrink-0', selectedConnectionId === account.id ? 'opacity-100' : 'opacity-0')} />
                    <AccountDisplay account={account} showScopeWarning />
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
      {error && <p className="text-xs text-destructive">{error}</p>}
      {onConnectAccount && (
        <Button variant="link" size="sm" className="h-auto p-0 text-xs text-seer" onClick={onConnectAccount}>
          <Link2 className="h-3 w-3 mr-1" />Connect another account
        </Button>
      )}
    </div>
  );
}
