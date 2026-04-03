/**
 * WorkflowTransferDialog Component
 *
 * Dialog for selecting and transferring workflows from personal workspace to a team.
 * This is a one-way operation that cannot be undone.
 */

import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, AlertCircle, ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { organizationApi } from '@/lib/organization-api';
import { workflowKeys } from '@/lib/query-keys';
import { toast } from '@/components/ui/sonner';
import type { Organization } from '@/types/organization';

interface TransferableWorkflow {
  workflow_id: string;
  name: string;
  description?: string;
}

interface WorkflowTransferDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  targetOrg: Organization;
  personalWorkflows: TransferableWorkflow[];
  onTransferComplete?: () => void;
}

/* eslint-disable max-lines-per-function */
export function WorkflowTransferDialog({
  open,
  onOpenChange,
  targetOrg,
  personalWorkflows,
  onTransferComplete,
}: WorkflowTransferDialogProps) {
  const queryClient = useQueryClient();
  const [selectedWorkflows, setSelectedWorkflows] = useState<string[]>([]);
  const [isTransferring, setIsTransferring] = useState(false);

  const handleToggleWorkflow = (workflowId: string) => {
    setSelectedWorkflows((prev) =>
      prev.includes(workflowId)
        ? prev.filter((id) => id !== workflowId)
        : [...prev, workflowId]
    );
  };

  const handleToggleAll = () => {
    if (selectedWorkflows.length === personalWorkflows.length) {
      setSelectedWorkflows([]);
    } else {
      setSelectedWorkflows(personalWorkflows.map((w) => w.workflow_id));
    }
  };

  const handleTransfer = async () => {
    if (selectedWorkflows.length === 0) {
      toast.error('Please select at least one workflow to transfer');
      return;
    }

    setIsTransferring(true);
    try {
      await organizationApi.transferWorkflows(targetOrg.id, selectedWorkflows);

      toast.success(
        `${selectedWorkflows.length} workflow${selectedWorkflows.length !== 1 ? 's' : ''} transferred to ${targetOrg.name}`
      );

      await queryClient.invalidateQueries({ queryKey: workflowKeys.list() });

      // Reset selection and close dialog
      setSelectedWorkflows([]);
      onOpenChange(false);
      onTransferComplete?.();
    } catch (error) {
      console.error('Failed to transfer workflows:', error);
      toast.error('Failed to transfer workflows');
    } finally {
      setIsTransferring(false);
    }
  };

  const handleClose = () => {
    if (!isTransferring) {
      setSelectedWorkflows([]);
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Transfer Workflows to Team</DialogTitle>
          <DialogDescription>
            Select workflows to transfer to {targetOrg.name}. This is a one-way ownership
            transfer and cannot be undone.
          </DialogDescription>
        </DialogHeader>

        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            Transferred workflows will be owned by the team. All team members will be able
            to access them based on their role.
          </AlertDescription>
        </Alert>

        <div className="max-h-[400px] overflow-y-auto space-y-2">
          {personalWorkflows.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              No personal workflows available to transfer
            </p>
          ) : (
            <>
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm text-muted-foreground">
                  {selectedWorkflows.length} of {personalWorkflows.length} selected
                </span>
                <Button variant="ghost" size="sm" onClick={handleToggleAll}>
                  {selectedWorkflows.length === personalWorkflows.length
                    ? 'Deselect All'
                    : 'Select All'}
                </Button>
              </div>

              {personalWorkflows.map((workflow) => {
                const isSelected = selectedWorkflows.includes(workflow.workflow_id);
                return (
                  <div
                    key={workflow.workflow_id}
                    className={cn(
                      'flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors',
                      isSelected
                        ? 'border-primary bg-primary/5'
                        : 'hover:bg-muted/50'
                    )}
                    onClick={() => handleToggleWorkflow(workflow.workflow_id)}
                  >
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => handleToggleWorkflow(workflow.workflow_id)}
                      onClick={(e) => e.stopPropagation()}
                    />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{workflow.name}</p>
                      {workflow.description && (
                        <p className="text-sm text-muted-foreground truncate">
                          {workflow.description}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isTransferring}>
            Cancel
          </Button>
          <Button
            onClick={handleTransfer}
            disabled={isTransferring || selectedWorkflows.length === 0}
          >
            {isTransferring ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Transferring...
              </>
            ) : (
              <>
                <ArrowRight className="h-4 w-4" />
                Transfer {selectedWorkflows.length} Workflow
                {selectedWorkflows.length !== 1 && 's'}
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
