/**
 * InvitationRow Component
 *
 * Displays a pending invitation with options to resend or revoke.
 */

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { MoreVertical, Mail, Clock, Loader2 } from 'lucide-react';
import { RoleBadge } from './RoleBadge';
import { useMembersStore } from '@/stores/membersStore';
import { toast } from '@/components/ui/sonner';
import type { Invitation } from '@/types/organization';

interface InvitationRowProps {
  invitation: Invitation;
  canManage: boolean;
}

export function InvitationRow({ invitation, canManage }: InvitationRowProps) {
  const [isResending, setIsResending] = useState(false);
  const [isRevoking, setIsRevoking] = useState(false);

  const resendInvitation = useMembersStore((s) => s.resendInvitation);
  const revokeInvitation = useMembersStore((s) => s.revokeInvitation);

  const handleResend = async () => {
    setIsResending(true);
    try {
      await resendInvitation(invitation.id);
      toast.success(`Invitation resent to ${invitation.email}`);
    } catch {
      toast.error('Failed to resend invitation');
    } finally {
      setIsResending(false);
    }
  };

  const handleRevoke = async () => {
    setIsRevoking(true);
    try {
      await revokeInvitation(invitation.id);
      toast.success('Invitation revoked');
    } catch {
      toast.error('Failed to revoke invitation');
    } finally {
      setIsRevoking(false);
    }
  };

  const isExpired = new Date(invitation.expiresAt) < new Date();
  const isLoading = isResending || isRevoking;

  const formatExpiryDate = () => {
    const date = new Date(invitation.expiresAt);
    if (isExpired) {
      return `Expired ${date.toLocaleDateString()}`;
    }
    return `Expires ${date.toLocaleDateString()}`;
  };

  return (
    <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/30 border-dashed">
      <div className="flex items-center gap-3 min-w-0">
        <div className="h-9 w-9 rounded-full bg-muted flex items-center justify-center">
          <Mail className="h-4 w-4 text-muted-foreground" />
        </div>
        <div className="min-w-0">
          <p className="font-medium text-sm truncate">{invitation.email}</p>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            <span>{formatExpiryDate()}</span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Badge
          variant="outline"
          className={
            isExpired
              ? 'bg-destructive/10 text-destructive border-destructive/20'
              : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'
          }
        >
          {isExpired ? 'Expired' : 'Pending'}
        </Badge>
        <RoleBadge role={invitation.role} />

        {canManage && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8" disabled={isLoading}>
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <MoreVertical className="h-4 w-4" />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={handleResend} disabled={isResending}>
                Resend invitation
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={handleRevoke}
                disabled={isRevoking}
                className="text-destructive focus:text-destructive"
              >
                Revoke invitation
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
      </div>
    </div>
  );
}
