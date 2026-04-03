import { useState, useEffect, useCallback } from 'react';
import { Check, ChevronsUpDown, AlertTriangle, Link2, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { getToolAccounts, initiateConnection, type ToolAccountInfo, type ToolAccountsResponse } from '@/lib/api-client';
import type { ClarificationQuestion } from '@/types/discovery';
import { useAuth } from '@clerk/clerk-react';

export interface AccountPickerQuestionProps {
  question: ClarificationQuestion;
  value: string | undefined;
  onChange: (connectionId: string, displayName?: string) => void;
  isLoading?: boolean;
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

function useAccountPicker(
  toolName: string | undefined,
  userId: string | null | undefined,
  selectedValue: string | undefined,
  onChange: (connectionId: string, displayName?: string) => void
) {
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [accountsData, setAccountsData] = useState<ToolAccountsResponse | null>(null);
  const [fetchError, setFetchError] = useState<string | null>(null);

  const fetchAccounts = useCallback(async (isRefresh = false) => {
    if (!toolName) {
      setAccountsData(null);
      return;
    }

    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }
    setFetchError(null);

    try {
      const response = await getToolAccounts(toolName);
      setAccountsData(response);
    } catch (err) {
      setFetchError(err instanceof Error ? err.message : 'Failed to fetch accounts');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [toolName]);

  useEffect(() => {
    fetchAccounts();
  }, [fetchAccounts]);

  // Auto-select when there's exactly one account and nothing selected
  useEffect(() => {
    if (accountsData && accountsData.accounts.length === 1 && !selectedValue) {
      const account = accountsData.accounts[0];
      onChange(String(account.id), account.display_name);
    }
  }, [accountsData, selectedValue, onChange]);

  const handleConnectAccount = useCallback(async () => {
    if (!toolName || !userId || !accountsData?.provider) return;

    try {
      const callbackUrl = window.location.href;
      const scopes = accountsData.required_scopes?.join(' ') || '';
      if (!scopes) {
        console.error('No required scopes available for tool:', toolName);
        return;
      }
      const response = await initiateConnection({
        userId,
        provider: accountsData.provider,
        scope: scopes,
        callbackUrl,
        integrationType: accountsData.provider,
        forceNewAccount: true,
      });
      if (response.redirectUrl) {
        window.open(response.redirectUrl, '_blank');
      }
    } catch (err) {
      console.error('Failed to initiate connection:', err);
    }
  }, [toolName, userId, accountsData]);

  return { loading, refreshing, accountsData, fetchError, refresh: () => fetchAccounts(true), handleConnectAccount };
}

export function AccountPickerQuestion({
  question,
  value,
  onChange,
  isLoading = false,
}: AccountPickerQuestionProps) {
  const [open, setOpen] = useState(false);
  const { userId } = useAuth();
  const toolName = question.tool_name;
  const { loading, refreshing, accountsData, fetchError, refresh, handleConnectAccount } = useAccountPicker(toolName, userId, value, onChange);

  const handleSelect = useCallback((account: ToolAccountInfo) => {
    onChange(String(account.id), account.display_name);
    setOpen(false);
  }, [onChange]);

  if (!toolName) {
    return (
      <div className="text-xs text-destructive">
        Account picker requires tool_name to be specified
      </div>
    );
  }

  if (loading && !accountsData) {
    return (
      <div className="text-xs text-muted-foreground animate-pulse">
        Loading accounts...
      </div>
    );
  }

  if (fetchError) {
    return (
      <div className="space-y-2">
        <div className="text-xs text-destructive">{fetchError}</div>
        <Button variant="outline" size="sm" onClick={refresh}>
          <RefreshCw className="h-3 w-3 mr-1" />
          Retry
        </Button>
      </div>
    );
  }

  if (accountsData && accountsData.accounts.length === 0) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
          <span>No connected account for this tool</span>
        </div>
        <Button variant="outline" size="sm" onClick={handleConnectAccount}>
          <Link2 className="h-3 w-3 mr-1" />
          Connect Account
        </Button>
      </div>
    );
  }

  if (!accountsData) return null;

  const selectedAccount = value
    ? accountsData.accounts.find((acc) => String(acc.id) === value)
    : undefined;

  return (
    <div className="space-y-2">
      <AccountSelectorDropdown
        open={open}
        onOpenChange={setOpen}
        selectedAccount={selectedAccount}
        accounts={accountsData.accounts}
        selectedValue={value}
        isLoading={isLoading}
        onSelect={handleSelect}
      />
      <AccountPickerActions
        onConnect={handleConnectAccount}
        onRefresh={refresh}
        isLoading={isLoading}
        refreshing={refreshing}
      />
    </div>
  );
}

function AccountSelectorDropdown({
  open,
  onOpenChange,
  selectedAccount,
  accounts,
  selectedValue,
  isLoading,
  onSelect,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedAccount: ToolAccountInfo | undefined;
  accounts: ToolAccountInfo[];
  selectedValue: string | undefined;
  isLoading: boolean;
  onSelect: (account: ToolAccountInfo) => void;
}) {
  return (
    <Popover open={open} onOpenChange={onOpenChange}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={isLoading}
          className={cn(
            'w-full justify-between font-normal text-left',
            !selectedAccount && 'text-muted-foreground'
          )}
        >
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
              {accounts.map((account) => (
                <CommandItem
                  key={account.id}
                  value={account.display_name}
                  onSelect={() => onSelect(account)}
                  className="flex items-center gap-2"
                >
                  <Check
                    className={cn(
                      'h-4 w-4 flex-shrink-0',
                      selectedValue === String(account.id) ? 'opacity-100' : 'opacity-0'
                    )}
                  />
                  <AccountDisplay account={account} showScopeWarning />
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}

function AccountPickerActions({
  onConnect,
  onRefresh,
  isLoading,
  refreshing,
}: {
  onConnect: () => void;
  onRefresh: () => void;
  isLoading: boolean;
  refreshing: boolean;
}) {
  return (
    <div className="flex items-center gap-2">
      <Button
        variant="link"
        size="sm"
        className="h-auto p-0 text-xs text-seer"
        onClick={onConnect}
        disabled={isLoading}
      >
        <Link2 className="h-3 w-3 mr-1" />
        Connect another account
      </Button>
      <Button
        variant="ghost"
        size="sm"
        className="h-auto p-0 text-xs"
        onClick={onRefresh}
        disabled={isLoading || refreshing}
      >
        <RefreshCw className={cn('h-3 w-3 mr-1', refreshing && 'animate-spin')} />
        Refresh
      </Button>
    </div>
  );
}
