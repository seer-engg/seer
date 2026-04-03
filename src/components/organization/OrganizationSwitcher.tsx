/**
 * OrganizationSwitcher Component
 *
 * Dropdown component for switching between personal workspace and team workspaces.
 * Displays in the sidebar and shows the current organization context.
 */

import { useState, forwardRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Building2, ChevronDown, Plus, User, Check, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useOrganizationStore } from '@/stores/organizationStore';
import { CreateTeamDialog } from './CreateTeamDialog';

interface OrgSwitcherTriggerProps {
  org: Organization;
  isSwitching: boolean;
  user?: { imageUrl?: string; fullName?: string | null } | null;
}

/**
 * Trigger button for the organization switcher dropdown.
 * Must use forwardRef for Radix's asChild to work properly.
 */
const OrgSwitcherTrigger = forwardRef<HTMLButtonElement, OrgSwitcherTriggerProps>(
  ({ org, isSwitching, user, ...props }, ref) => {
    return (
      <Button
        ref={ref}
        variant="ghost"
        className="w-full justify-between px-2 h-9 text-sm font-normal"
        disabled={isSwitching}
        {...props}
      >
        <div className="flex items-center gap-2 min-w-0">
          {isSwitching ? (
            <Loader2 className="h-4 w-4 shrink-0 animate-spin" />
          ) : org.type === 'personal' && user?.imageUrl ? (
            <Avatar className="h-4 w-4 shrink-0">
              <AvatarImage src={user.imageUrl} alt={user.fullName || 'User'} />
              <AvatarFallback><User className="h-3 w-3" /></AvatarFallback>
            </Avatar>
          ) : org.type === 'personal' ? (
            <User className="h-4 w-4 shrink-0" />
          ) : (
            <Building2 className="h-4 w-4 shrink-0" />
          )}
          <span className="truncate">{org.type === 'personal' ? (user?.fullName || 'Personal') : org.name}</span>
        </div>
        <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
      </Button>
    );
  }
);
OrgSwitcherTrigger.displayName = 'OrgSwitcherTrigger';

interface Organization {
  id: number;
  name: string;
  type: 'personal' | 'team';
  role: string;
  memberCount?: number;
}

export function OrganizationSwitcher() {
  const navigate = useNavigate();
  const { user } = useUser();
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);

  const organizations = useOrganizationStore((s) => s.organizations);
  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const switchOrganization = useOrganizationStore((s) => s.switchOrganization);
  const isLoading = useOrganizationStore((s) => s.isLoading);
  const isSwitching = useOrganizationStore((s) => s.isSwitching);
  const isInitialized = useOrganizationStore((s) => s.isInitialized);

  if (!isInitialized || isLoading) {
    return <div className="px-2 py-1.5"><Skeleton className="h-9 w-full" /></div>;
  }

  if (!currentOrganization) return null;

  const personalOrg = organizations.find((o) => o.type === 'personal');
  const teamOrgs = organizations.filter((o) => o.type === 'team');

  const handleSwitch = async (orgId: number) => {
    if (orgId === currentOrganization.id) return;
    try { await switchOrganization(orgId); } catch { /* handled */ }
  };

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <OrgSwitcherTrigger org={currentOrganization as Organization} isSwitching={isSwitching} user={user} />
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[220px]">
          <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">Workspaces</DropdownMenuLabel>
          {personalOrg && (
            <DropdownMenuItem onClick={() => handleSwitch(personalOrg.id)} className={cn('flex items-center gap-2 cursor-pointer', currentOrganization.id === personalOrg.id && 'bg-accent')}>
              {user?.imageUrl ? (
                <Avatar className="h-4 w-4 shrink-0">
                  <AvatarImage src={user.imageUrl} alt={user.fullName || 'User'} />
                  <AvatarFallback><User className="h-3 w-3" /></AvatarFallback>
                </Avatar>
              ) : (
                <User className="h-4 w-4 shrink-0" />
              )}
              <span className="flex-1 truncate">{user?.fullName || 'Personal'}</span>
              {currentOrganization.id === personalOrg.id && <Check className="h-4 w-4 shrink-0 text-primary" />}
            </DropdownMenuItem>
          )}
          {teamOrgs.length > 0 && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuLabel className="text-xs text-muted-foreground font-normal">Teams</DropdownMenuLabel>
              {teamOrgs.map((org) => (
                <DropdownMenuItem key={org.id} onClick={() => handleSwitch(org.id)} className={cn('flex items-center gap-2 cursor-pointer', currentOrganization.id === org.id && 'bg-accent')}>
                  <Building2 className="h-4 w-4 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <span className="block truncate">{org.name}</span>
                    <span className="text-xs text-muted-foreground">{org.role}{org.memberCount !== undefined && ` • ${org.memberCount} members`}</span>
                  </div>
                  {currentOrganization.id === org.id && <Check className="h-4 w-4 shrink-0 text-primary" />}
                </DropdownMenuItem>
              ))}
            </>
          )}
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={() => setIsCreateDialogOpen(true)} className="flex items-center gap-2 cursor-pointer">
            <Plus className="h-4 w-4 shrink-0" /><span>Create Team</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <CreateTeamDialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen} onSuccess={() => { setIsCreateDialogOpen(false); navigate('/settings?tab=team'); }} />
    </>
  );
}
