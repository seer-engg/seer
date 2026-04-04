/**
 * InviteMemberDialog Component
 *
 * Dialog for inviting new members to a team organization.
 */

import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Info } from 'lucide-react';
import { useMembersStore } from '@/stores/membersStore';
import { toast } from '@/components/ui/sonner';
import type { OrganizationRole } from '@/types/organization';
import { getAssignableRoles, getRoleDisplayName, getRoleDescription } from '@/types/organization';

interface InviteMemberDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentUserRole: OrganizationRole | null;
}

/* eslint-disable max-lines-per-function */
export function InviteMemberDialog({
  open,
  onOpenChange,
  currentUserRole,
}: InviteMemberDialogProps) {
  const [email, setEmail] = useState('');
  const [role, setRole] = useState<OrganizationRole>('user');

  const inviteMember = useMembersStore((s) => s.inviteMember);
  const isInviting = useMembersStore((s) => s.isInviting);

  const assignableRoles = getAssignableRoles(currentUserRole);

  const handleInvite = async () => {
    if (!email.trim()) return;

    try {
      await inviteMember(email.trim(), role);
      toast.success(`Invitation sent to ${email}`);
      handleClose();
    } catch {
      toast.error('Failed to send invitation');
    }
  };

  const handleClose = () => {
    if (!isInviting) {
      setEmail('');
      setRole('user');
      onOpenChange(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && email.trim() && !isInviting) {
      e.preventDefault();
      handleInvite();
    }
  };

  const isValidEmail = (value: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Invite Team Member</DialogTitle>
          <DialogDescription>
            Send an invitation to join your team workspace.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          <div className="space-y-2">
            <Label htmlFor="invite-email">Email Address</Label>
            <Input
              id="invite-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="colleague@company.com"
              disabled={isInviting}
              autoFocus
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="invite-role">Role</Label>
            <Select
              value={role}
              onValueChange={(value) => setRole(value as OrganizationRole)}
              disabled={isInviting}
            >
              <SelectTrigger id="invite-role">
                <SelectValue placeholder="Select a role" />
              </SelectTrigger>
              <SelectContent>
                {assignableRoles.map((r) => (
                  <SelectItem key={r} value={r}>
                    <div className="flex flex-col items-start">
                      <span>{getRoleDisplayName(r)}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              {getRoleDescription(role)}
            </p>
          </div>

          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription className="ml-2">
              The invitation will be sent via email. It expires in 7 days.
            </AlertDescription>
          </Alert>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isInviting}>
            Cancel
          </Button>
          <Button
            onClick={handleInvite}
            disabled={!email.trim() || !isValidEmail(email) || isInviting}
          >
            {isInviting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Sending...
              </>
            ) : (
              'Send Invitation'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
