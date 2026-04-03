/**
 * ApprovalsSection Component
 *
 * Panel for admins/owners to review and manage pending workflow approvals
 * from consultants.
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { ClipboardCheck, Check, X, Loader2, Clock, FileText } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useOrganizationStore } from '@/stores/organizationStore';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { canReviewApprovals } from '@/types/organization';
import { toast } from '@/components/ui/sonner';
import type { WorkflowApproval } from '@/types/organization';

interface ReviewDialogState {
  open: boolean;
  approval: WorkflowApproval | null;
  action: 'approved' | 'rejected' | null;
}

/* eslint-disable max-lines-per-function, complexity */
export function ApprovalsSection() {
  const [reviewDialog, setReviewDialog] = useState<ReviewDialogState>({
    open: false,
    approval: null,
    action: null,
  });
  const [reviewNotes, setReviewNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const getCurrentRole = useOrganizationStore((s) => s.getCurrentRole);

  const isLoading = useApprovalsStore((s) => s.isLoading);
  const fetchApprovals = useApprovalsStore((s) => s.fetchApprovals);
  const reviewApproval = useApprovalsStore((s) => s.reviewApproval);
  const getPendingApprovals = useApprovalsStore((s) => s.getPendingApprovals);

  const role = getCurrentRole();
  const canReview = canReviewApprovals(role);
  const pendingApprovals = getPendingApprovals();
  const orgId = currentOrganization?.id;

  // Fetch approvals when component mounts or org changes
  useEffect(() => {
    if (orgId && canReview) {
      fetchApprovals(orgId);
    }
  }, [orgId, canReview, fetchApprovals]);

  // Don't render if user can't review approvals or if in personal workspace
  if (!canReview || !currentOrganization || currentOrganization.type === 'personal') {
    return null;
  }

  // Don't show if no pending approvals
  if (!isLoading && pendingApprovals.length === 0) {
    return null;
  }

  const openReviewDialog = (approval: WorkflowApproval, action: 'approved' | 'rejected') => {
    setReviewDialog({ open: true, approval, action });
    setReviewNotes('');
  };

  const closeReviewDialog = () => {
    setReviewDialog({ open: false, approval: null, action: null });
    setReviewNotes('');
  };

  const handleSubmitReview = async () => {
    if (!reviewDialog.approval || !reviewDialog.action) return;

    setIsSubmitting(true);
    try {
      await reviewApproval(
        reviewDialog.approval.id,
        reviewDialog.action,
        reviewNotes || undefined
      );
      toast.success(
        `Workflow ${reviewDialog.action === 'approved' ? 'approved' : 'rejected'}`
      );
      closeReviewDialog();
    } catch (error) {
      console.error('Failed to review approval:', error);
      toast.error('Failed to submit review');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <ClipboardCheck className="h-5 w-5" />
            Pending Approvals
            {pendingApprovals.length > 0 && (
              <Badge variant="secondary">{pendingApprovals.length}</Badge>
            )}
          </CardTitle>
          <CardDescription>
            Workflows created by consultants awaiting your review
          </CardDescription>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : pendingApprovals.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-6">
              No pending approvals
            </p>
          ) : (
            <div className="space-y-3">
              {pendingApprovals.map((approval) => (
                <div
                  key={approval.id}
                  className="flex items-center justify-between p-4 rounded-lg border"
                >
                  <div className="flex items-start gap-3">
                    <div className="h-9 w-9 rounded-md bg-amber-500/10 flex items-center justify-center shrink-0">
                      <Clock className="h-4 w-4 text-amber-500" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground shrink-0" />
                        <p className="font-medium truncate">{approval.workflowName}</p>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Requested by {approval.requestedBy.name} &middot;{' '}
                        {formatDistanceToNow(new Date(approval.requestedAt))} ago
                      </p>
                    </div>
                  </div>

                  <div className="flex gap-2 shrink-0 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => openReviewDialog(approval, 'rejected')}
                    >
                      <X className="h-4 w-4" />
                      Reject
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => openReviewDialog(approval, 'approved')}
                    >
                      <Check className="h-4 w-4" />
                      Approve
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Review Dialog */}
      <Dialog open={reviewDialog.open} onOpenChange={(open) => !open && closeReviewDialog()}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {reviewDialog.action === 'approved' ? 'Approve' : 'Reject'} Workflow
            </DialogTitle>
            <DialogDescription>
              {reviewDialog.action === 'approved'
                ? 'This workflow will become visible to the team after approval.'
                : 'The consultant will be notified that their workflow was rejected.'}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div>
              <Label className="text-sm font-medium">Workflow</Label>
              <p className="text-sm text-muted-foreground">
                {reviewDialog.approval?.workflowName}
              </p>
            </div>

            <div>
              <Label className="text-sm font-medium">Requested by</Label>
              <p className="text-sm text-muted-foreground">
                {reviewDialog.approval?.requestedBy.name} (
                {reviewDialog.approval?.requestedBy.email})
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="review-notes">
                {reviewDialog.action === 'rejected' ? 'Feedback (recommended)' : 'Notes (optional)'}
              </Label>
              <Textarea
                id="review-notes"
                value={reviewNotes}
                onChange={(e) => setReviewNotes(e.target.value)}
                placeholder={
                  reviewDialog.action === 'rejected'
                    ? 'Explain why this workflow was rejected...'
                    : 'Add any notes about this approval...'
                }
                rows={3}
              />
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={closeReviewDialog} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button
              variant={reviewDialog.action === 'rejected' ? 'destructive' : 'default'}
              onClick={handleSubmitReview}
              disabled={isSubmitting}
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Submitting...
                </>
              ) : reviewDialog.action === 'approved' ? (
                <>
                  <Check className="h-4 w-4" />
                  Approve Workflow
                </>
              ) : (
                <>
                  <X className="h-4 w-4" />
                  Reject Workflow
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
