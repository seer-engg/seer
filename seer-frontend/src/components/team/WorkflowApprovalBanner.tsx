/**
 * WorkflowApprovalBanner Component
 *
 * Banner displayed in the workflow editor for consultant-created workflows.
 * Shows the current approval status and allows requesting approval.
 */

import { useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertCircle, Clock, XCircle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useOrganizationStore } from '@/stores/organizationStore';
import { useApprovalsStore } from '@/stores/approvalsStore';
import { toast } from '@/components/ui/sonner';
import type { ApprovalStatus } from '@/types/organization';

interface WorkflowApprovalBannerProps {
  workflowId: string;
  approvalStatus?: ApprovalStatus;
  className?: string;
}

export function WorkflowApprovalBanner({
  workflowId,
  approvalStatus,
  className,
}: WorkflowApprovalBannerProps) {
  const [isRequesting, setIsRequesting] = useState(false);

  const getCurrentRole = useOrganizationStore((s) => s.getCurrentRole);
  const isCurrentOrgPersonal = useOrganizationStore((s) => s.isCurrentOrgPersonal);
  const requestApproval = useApprovalsStore((s) => s.requestApproval);

  const role = getCurrentRole();
  const isConsultant = role === 'consultant';
  const isPersonal = isCurrentOrgPersonal();

  // Only show for consultants in team context
  if (!isConsultant || isPersonal) {
    return null;
  }

  // Already approved - don't show banner
  if (approvalStatus === 'approved') {
    return null;
  }

  const handleRequestApproval = async () => {
    setIsRequesting(true);
    try {
      await requestApproval(workflowId);
      toast.success('Approval requested');
    } catch (error) {
      console.error('Failed to request approval:', error);
      toast.error('Failed to request approval');
    } finally {
      setIsRequesting(false);
    }
  };

  const getStatusConfig = () => {
    switch (approvalStatus) {
      case 'pending':
        return {
          icon: Clock,
          title: 'Pending Approval',
          description: 'This workflow is waiting for admin approval before it can be published.',
          variant: 'warning' as const,
          bgClass: 'bg-amber-500/10 border-amber-500/20',
          iconClass: 'text-amber-500',
        };
      case 'rejected':
        return {
          icon: XCircle,
          title: 'Approval Rejected',
          description: 'This workflow was rejected. Please review the feedback and resubmit.',
          variant: 'destructive' as const,
          bgClass: 'bg-destructive/10 border-destructive/20',
          iconClass: 'text-destructive',
        };
      default:
        // draft or not submitted
        return {
          icon: AlertCircle,
          title: 'Approval Required',
          description:
            'As a consultant, your workflows need approval before they are visible to the team.',
          variant: 'default' as const,
          bgClass: 'bg-primary/10 border-primary/20',
          iconClass: 'text-primary',
        };
    }
  };

  const config = getStatusConfig();
  const Icon = config.icon;

  return (
    <Alert className={cn(config.bgClass, className)}>
      <Icon className={cn('h-4 w-4', config.iconClass)} />
      <AlertTitle>{config.title}</AlertTitle>
      <AlertDescription className="flex items-center justify-between">
        <span>{config.description}</span>
        {approvalStatus !== 'pending' && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleRequestApproval}
            disabled={isRequesting}
            className="ml-4 shrink-0"
          >
            {isRequesting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Requesting...
              </>
            ) : approvalStatus === 'rejected' ? (
              'Resubmit for Approval'
            ) : (
              'Request Approval'
            )}
          </Button>
        )}
      </AlertDescription>
    </Alert>
  );
}
