/**
 * WorkflowTransferSection Component
 *
 * Card component that shows the option to transfer workflows from personal workspace
 * to the current team. Fetches personal workspace workflows and shows the transfer dialog.
 */

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight, FolderInput, Loader2 } from 'lucide-react';
import { backendApiClient } from '@/lib/api-client';
import { useOrganizationStore } from '@/stores/organizationStore';
import { WorkflowTransferDialog } from './WorkflowTransferDialog';
import type { Organization } from '@/types/organization';

interface PersonalWorkflow {
  workflow_id: string;
  name: string;
}

interface WorkflowListResponse {
  items: PersonalWorkflow[];
}

interface WorkflowTransferSectionProps {
  targetOrg: Organization;
}

export function WorkflowTransferSection({ targetOrg }: WorkflowTransferSectionProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [personalWorkflows, setPersonalWorkflows] = useState<PersonalWorkflow[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const getPersonalOrg = useOrganizationStore((s) => s.getPersonalOrg);

  // Don't show on personal workspace
  if (targetOrg.type === 'personal') {
    return null;
  }

  const personalOrg = getPersonalOrg();

  // Fetch personal workflows when the dialog is about to open
  const handleOpenDialog = async () => {
    if (!personalOrg) {
      setError('Could not find personal workspace');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Fetch workflows for the personal workspace
      // The backend should support an org_id query parameter
      const response = await backendApiClient.request<WorkflowListResponse>(
        `/api/v1/workflows?org_id=${personalOrg.id}`,
        { method: 'GET' }
      );
      setPersonalWorkflows(response.items || []);
      setIsDialogOpen(true);
    } catch (err) {
      console.error('Failed to fetch personal workflows:', err);
      setError('Failed to load personal workflows');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTransferComplete = () => {
    // Clear the cached personal workflows after transfer
    setPersonalWorkflows([]);
  };

  return (
    <>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderInput className="h-5 w-5" />
            Transfer Workflows
          </CardTitle>
          <CardDescription>
            Move workflows from your personal workspace to this team
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">
                Transfer your personal workflows to share them with team members
              </p>
              {error && (
                <p className="text-sm text-destructive mt-1">{error}</p>
              )}
            </div>
            <Button
              variant="outline"
              onClick={handleOpenDialog}
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading...
                </>
              ) : (
                <>
                  <ArrowRight className="h-4 w-4" />
                  Transfer Workflows
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {personalOrg && (
        <WorkflowTransferDialog
          open={isDialogOpen}
          onOpenChange={setIsDialogOpen}
          targetOrg={targetOrg}
          personalWorkflows={personalWorkflows}
          onTransferComplete={handleTransferComplete}
        />
      )}
    </>
  );
}
