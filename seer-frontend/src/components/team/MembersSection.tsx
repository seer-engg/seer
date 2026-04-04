/**
 * MembersSection Component
 *
 * Displays the members list and pending invitations for a team.
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { UserPlus, Users } from 'lucide-react';
import { MemberRow } from './MemberRow';
import { InvitationRow } from './InvitationRow';
import { InviteMemberDialog } from './InviteMemberDialog';
import { useMembersStore } from '@/stores/membersStore';
import { useOrganizationStore } from '@/stores/organizationStore';

/* eslint-disable max-lines-per-function */
export function MembersSection() {
  const [isInviteOpen, setIsInviteOpen] = useState(false);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const canInviteMembers = useOrganizationStore((s) => s.canInviteMembers);
  const canManageMembers = useOrganizationStore((s) => s.canManageMembers);
  const getCurrentRole = useOrganizationStore((s) => s.getCurrentRole);

  const members = useMembersStore((s) => s.members);
  const invitations = useMembersStore((s) => s.invitations);
  const isLoading = useMembersStore((s) => s.isLoading);
  const fetchMembers = useMembersStore((s) => s.fetchMembers);
  const fetchInvitations = useMembersStore((s) => s.fetchInvitations);

  useEffect(() => {
    if (currentOrganization?.id) {
      fetchMembers(currentOrganization.id);
      fetchInvitations(currentOrganization.id);
    }
  }, [currentOrganization?.id, fetchMembers, fetchInvitations]);

  const pendingInvitations = invitations.filter((i) => i.status === 'pending');
  const currentRole = getCurrentRole();

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Members
          </CardTitle>
          <CardDescription>
            {members.length} member{members.length !== 1 && 's'}
            {pendingInvitations.length > 0 && (
              <span className="text-amber-600 dark:text-amber-400">
                {' '}• {pendingInvitations.length} pending invitation{pendingInvitations.length !== 1 && 's'}
              </span>
            )}
          </CardDescription>
        </div>
        {canInviteMembers() && (
          <Button size="sm" onClick={() => setIsInviteOpen(true)}>
            <UserPlus className="h-4 w-4" />
            <span className="hidden sm:inline">Invite</span>
          </Button>
        )}
      </CardHeader>

      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            <MemberRowSkeleton />
            <MemberRowSkeleton />
            <MemberRowSkeleton />
          </div>
        ) : (
          <div className="space-y-3">
            {/* Active Members */}
            {members.map((member) => (
              <MemberRow
                key={member.id}
                member={member}
                canManage={canManageMembers()}
                currentUserRole={currentRole}
              />
            ))}

            {/* Pending Invitations */}
            {pendingInvitations.length > 0 && (
              <>
                <div className="pt-2">
                  <p className="text-xs text-muted-foreground font-medium uppercase tracking-wider mb-2">
                    Pending Invitations
                  </p>
                </div>
                {pendingInvitations.map((invitation) => (
                  <InvitationRow
                    key={invitation.id}
                    invitation={invitation}
                    canManage={canManageMembers()}
                  />
                ))}
              </>
            )}

            {members.length === 0 && pendingInvitations.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Users className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No team members yet</p>
                {canInviteMembers() && (
                  <Button
                    variant="link"
                    className="mt-2"
                    onClick={() => setIsInviteOpen(true)}
                  >
                    Invite your first team member
                  </Button>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>

      <InviteMemberDialog
        open={isInviteOpen}
        onOpenChange={setIsInviteOpen}
        currentUserRole={currentRole}
      />
    </Card>
  );
}

function MemberRowSkeleton() {
  return (
    <div className="flex items-center justify-between p-3 rounded-lg border">
      <div className="flex items-center gap-3">
        <Skeleton className="h-9 w-9 rounded-full" />
        <div className="space-y-1.5">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-3 w-40" />
        </div>
      </div>
      <Skeleton className="h-5 w-16" />
    </div>
  );
}
