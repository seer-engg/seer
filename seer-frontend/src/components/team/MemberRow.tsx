/**
 * MemberRow Component
 *
 * Displays a single team member with role management options.
 */

import { useState } from 'react';
import { useCurrentUser } from '@/hooks/useAuthProvider';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
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
import { MoreVertical, Loader2 } from 'lucide-react';
import { RoleBadge } from './RoleBadge';
import { useMembersStore } from '@/stores/membersStore';
import { toast } from '@/components/ui/sonner';
import type { Member, OrganizationRole } from '@/types/organization';
import { getAssignableRoles, getRoleDisplayName } from '@/types/organization';

interface MemberRowProps {
  member: Member;
  canManage: boolean;
  currentUserRole: OrganizationRole | null;
}

/* eslint-disable max-lines-per-function */
export function MemberRow({ member, canManage, currentUserRole }: MemberRowProps) {
  const { user } = useCurrentUser();
  const [isUpdating, setIsUpdating] = useState(false);
  const [isRemoving, setIsRemoving] = useState(false);
  const [showRemoveDialog, setShowRemoveDialog] = useState(false);

  const updateMemberRole = useMembersStore((s) => s.updateMemberRole);
  const removeMember = useMembersStore((s) => s.removeMember);

  const isCurrentUser = member.email === user?.primaryEmailAddress?.emailAddress;
  const isOwner = member.role === 'owner';
  const canModify = canManage && !isOwner && !isCurrentUser;

  const assignableRoles = getAssignableRoles(currentUserRole);

  const handleRoleChange = async (newRole: OrganizationRole) => {
    if (newRole === member.role) return;

    setIsUpdating(true);
    try {
      await updateMemberRole(member.userId, newRole);
      toast.success(`Updated ${member.firstName || member.email}'s role to ${getRoleDisplayName(newRole)}`);
    } catch {
      toast.error('Failed to update role');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleRemove = async () => {
    setIsRemoving(true);
    try {
      await removeMember(member.userId);
      toast.success(`Removed ${member.firstName || member.email} from the team`);
      setShowRemoveDialog(false);
    } catch {
      toast.error('Failed to remove member');
    } finally {
      setIsRemoving(false);
    }
  };

  const getInitials = () => {
    if (member.firstName) {
      return member.firstName[0].toUpperCase();
    }
    return member.email[0].toUpperCase();
  };

  const getDisplayName = () => {
    if (member.firstName || member.lastName) {
      return `${member.firstName || ''} ${member.lastName || ''}`.trim();
    }
    return member.email;
  };

  return (
    <>
      <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
        <div className="flex items-center gap-3 min-w-0">
          <Avatar className="h-9 w-9">
            <AvatarImage src={member.avatarUrl} alt={getDisplayName()} />
            <AvatarFallback className="text-sm">{getInitials()}</AvatarFallback>
          </Avatar>
          <div className="min-w-0">
            <p className="font-medium text-sm truncate">
              {getDisplayName()}
              {isCurrentUser && (
                <span className="text-muted-foreground font-normal"> (you)</span>
              )}
            </p>
            <p className="text-xs text-muted-foreground truncate">{member.email}</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <RoleBadge role={member.role} />

          {canModify && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-8 w-8" disabled={isUpdating}>
                  {isUpdating ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <MoreVertical className="h-4 w-4" />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuLabel>Change Role</DropdownMenuLabel>
                {assignableRoles.map((role) => (
                  <DropdownMenuItem
                    key={role}
                    onClick={() => handleRoleChange(role)}
                    disabled={member.role === role}
                    className={member.role === role ? 'opacity-50' : ''}
                  >
                    {getRoleDisplayName(role)}
                  </DropdownMenuItem>
                ))}
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  className="text-destructive focus:text-destructive"
                  onClick={() => setShowRemoveDialog(true)}
                >
                  Remove from team
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </div>

      <AlertDialog open={showRemoveDialog} onOpenChange={setShowRemoveDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove team member</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to remove {getDisplayName()} from the team?
              They will lose access to all team resources.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isRemoving}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleRemove}
              disabled={isRemoving}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isRemoving ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Removing...
                </>
              ) : (
                'Remove'
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
