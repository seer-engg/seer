/**
 * CreateTeamDialog Component
 *
 * Dialog for creating a new team organization.
 * Requires an active subscription - the subscription is transferred to the new team.
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, AlertCircle, Info } from 'lucide-react';
import { useOrganizationStore } from '@/stores/organizationStore';
import { useSubscriptionStore } from '@/stores/subscriptionStore';
import { toast } from '@/components/ui/sonner';

interface CreateTeamDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSuccess?: () => void;
}

/* eslint-disable max-lines-per-function */
export function CreateTeamDialog({ open, onOpenChange, onSuccess }: CreateTeamDialogProps) {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  const createOrganization = useOrganizationStore((s) => s.createOrganization);
  const subscription = useSubscriptionStore((s) => s.subscription);

  // Check if user has an active subscription
  const hasActiveSubscription =
    subscription?.status === 'active' || subscription?.status === 'trialing';

  const handleCreate = async () => {
    if (!name.trim() || !hasActiveSubscription) return;

    setIsCreating(true);
    try {
      await createOrganization(name.trim());
      toast.success('Team created successfully');
      setName('');
      onOpenChange(false);
      onSuccess?.();
    } catch (error) {
      console.error('Failed to create team:', error);
      toast.error('Failed to create team');
    } finally {
      setIsCreating(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && name.trim() && hasActiveSubscription && !isCreating) {
      e.preventDefault();
      handleCreate();
    }
  };

  const handleClose = () => {
    if (!isCreating) {
      setName('');
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Create a Team</DialogTitle>
          <DialogDescription>
            Create a team workspace to collaborate with others.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {!hasActiveSubscription ? (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="ml-2">
                You need an active subscription to create a team.{' '}
                <Button
                  variant="link"
                  className="h-auto p-0 text-destructive underline"
                  onClick={() => {
                    onOpenChange(false);
                    navigate('/settings/billing');
                  }}
                >
                  Upgrade now
                </Button>
              </AlertDescription>
            </Alert>
          ) : (
            <>
              <div className="space-y-2">
                <Label htmlFor="team-name">Team Name</Label>
                <Input
                  id="team-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Acme Inc."
                  disabled={isCreating}
                  autoFocus
                />
              </div>

              <Alert>
                <Info className="h-4 w-4" />
                <AlertDescription className="ml-2">
                  Your <strong>{subscription?.tier || 'current'}</strong> subscription will be
                  transferred to this team. Your personal workspace will revert to the Free tier.
                </AlertDescription>
              </Alert>
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isCreating}>
            Cancel
          </Button>
          <Button
            onClick={handleCreate}
            disabled={!hasActiveSubscription || !name.trim() || isCreating}
          >
            {isCreating ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Creating...
              </>
            ) : (
              'Create Team'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
