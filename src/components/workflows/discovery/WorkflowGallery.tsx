import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Loader2, Download, FileEdit, Copy, Trash2, Upload } from 'lucide-react';

import { useWorkflowStore, type WorkflowListItem } from '@/stores/workflowStore';
import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';
import { generateWorkflowName } from '@/lib/workflow-names';
import { toast } from '@/components/ui/sonner';
import { Skeleton } from '@/components/ui/skeleton';
import { Button } from '@/components/ui/button';
import { IntegrationIcon } from '@/components/workflows/build/UnifiedBuildItem';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Input } from '@/components/ui/input';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { cn } from '@/lib/utils';

import { DeleteWorkflowDialog } from './DeleteWorkflowDialog';
import { WorkflowImportDialog } from '@/components/workflows/dialogs/WorkflowImportDialog';
import { BackendAPIError } from '@/lib/api-client';
import { useWorkflowListQuery } from '@/hooks/useWorkflowQueries';

export interface WorkflowGalleryProps {
  className?: string;
}

interface WorkflowRowProps {
  workflow: WorkflowListItem;
  renamingId: string | null;
  renameValue: string;
  setRenameValue: (v: string) => void;
  onNavigate: (id: string) => void;
  onRenameSave: (id: string) => void;
  onRenameCancel: () => void;
  onRenameStart: (w: WorkflowListItem) => void;
  onExport: (id: string) => void;
  onDuplicate: (id: string) => void;
  onDelete: (w: { id: string; name: string }) => void;
}

function WorkflowRow({
  workflow, renamingId, renameValue, setRenameValue,
  onNavigate, onRenameSave, onRenameCancel, onRenameStart,
  onExport, onDuplicate, onDelete,
}: WorkflowRowProps) {
  return (
    <TableRow>
      <TableCell className="text-left">
        {renamingId === workflow.workflow_id ? (
          <Input
            autoFocus
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onBlur={() => onRenameSave(workflow.workflow_id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') onRenameSave(workflow.workflow_id);
              if (e.key === 'Escape') onRenameCancel();
            }}
            className="h-7 text-sm w-60"
          />
        ) : (
          <button className="text-sm font-medium hover:underline text-left" onClick={() => onNavigate(workflow.workflow_id)}>
            {workflow.name}
          </button>
        )}
      </TableCell>
      <TableCell>
        <div className="flex flex-wrap gap-1.5">
          {(workflow.integrations ?? []).map((integration) => (
            <Tooltip key={integration}>
              <TooltipTrigger asChild>
                <span className="capitalize"><IntegrationIcon integrationType={integration} /></span>
              </TooltipTrigger>
              <TooltipContent>{useIntegrationMetadataStore.getState().getDisplayName(integration)}</TooltipContent>
            </Tooltip>
          ))}
        </div>
      </TableCell>
      <TableCell>
        <div className="flex gap-0.5">
          <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onExport(workflow.workflow_id)}><Download className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Export</TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onRenameStart(workflow)}><FileEdit className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Rename</TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => onDuplicate(workflow.workflow_id)}><Copy className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Duplicate</TooltipContent></Tooltip>
          <Tooltip><TooltipTrigger asChild><Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => onDelete({ id: workflow.workflow_id, name: workflow.name })}><Trash2 className="h-4 w-4" /></Button></TooltipTrigger><TooltipContent>Delete</TooltipContent></Tooltip>
        </div>
      </TableCell>
    </TableRow>
  );
}

function useWorkflowActions() {
  const navigate = useNavigate();
  const createWorkflow = useWorkflowStore((state) => state.createWorkflow);
  const updateWorkflowMetadata = useWorkflowStore((state) => state.updateWorkflowMetadata);
  const deleteWorkflow = useWorkflowStore((state) => state.deleteWorkflow);
  const exportWorkflow = useWorkflowStore((state) => state.exportWorkflow);
  const duplicateWorkflow = useWorkflowStore((state) => state.duplicateWorkflow);
  const importWorkflow = useWorkflowStore((state) => state.importWorkflow);

  const handleCreateNew = useCallback(async (isCreating: boolean, setIsCreating: (v: boolean) => void) => {
    if (isCreating) return;
    setIsCreating(true);
    try {
      const workflow = await createWorkflow(generateWorkflowName(), { nodes: [], edges: [] });
      navigate(`/workflows/${workflow.workflow_id}`, { replace: true });
    } catch {
      toast.error('Failed to create workflow');
    } finally {
      setIsCreating(false);
    }
  }, [createWorkflow, navigate]);

  const handleRenameSave = useCallback(async (workflowId: string, renameValue: string, setRenamingId: (v: string | null) => void) => {
    const trimmed = renameValue.trim();
    if (!trimmed) { setRenamingId(null); return; }
    try {
      await updateWorkflowMetadata(workflowId, { name: trimmed });
      toast.success('Workflow renamed');
    } catch { toast.error('Failed to rename workflow'); }
    setRenamingId(null);
  }, [updateWorkflowMetadata]);

  const handleDeleteConfirm = useCallback(async (deleteTarget: { id: string } | null, setIsDeleting: (v: boolean) => void, setDeleteTarget: (v: null) => void) => {
    if (!deleteTarget) return;
    setIsDeleting(true);
    try {
      await deleteWorkflow(deleteTarget.id);
      toast.success('Workflow deleted');
    } catch { toast.error('Failed to delete workflow'); }
    finally { setIsDeleting(false); setDeleteTarget(null); }
  }, [deleteWorkflow]);

  const handleExport = useCallback(async (workflowId: string) => {
    try { await exportWorkflow(workflowId); toast.success('Workflow exported'); }
    catch { toast.error('Failed to export workflow'); }
  }, [exportWorkflow]);

  const handleDuplicate = useCallback(async (workflowId: string) => {
    try {
      const workflow = await duplicateWorkflow(workflowId);
      toast.success('Workflow duplicated');
      navigate(`/workflows/${workflow.workflow_id}`);
    } catch { toast.error('Failed to duplicate workflow'); }
  }, [duplicateWorkflow, navigate]);

  const handleImport = useCallback(async (file: File, options: { name?: string; importTriggers: boolean }) => {
    try {
      const result = await importWorkflow(file, options);
      navigate(`/workflows/${result.workflow_id}`, { replace: true });
      toast.success(`Workflow "${result.name}" imported successfully`);
    } catch (error) {
      console.error('Failed to import workflow:', error);
      toast.error(error instanceof BackendAPIError ? `Failed to import workflow: ${error.message}` : 'Failed to import workflow');
      throw error;
    }
  }, [importWorkflow, navigate]);

  return { navigate, handleCreateNew, handleRenameSave, handleDeleteConfirm, handleExport, handleDuplicate, handleImport };
}

export function WorkflowGallery({ className }: WorkflowGalleryProps) {
  const { data: workflows = [], isLoading, isError, isSuccess } = useWorkflowListQuery();
  const actions = useWorkflowActions();

  const [isCreating, setIsCreating] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<{ id: string; name: string } | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');

  const showSkeleton = isLoading && !isSuccess;

  return (
    <div className={cn('w-full border-t border-border/50 pt-6', className)}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-semibold text-foreground">Your Workflows</h2>
        <div className="flex gap-2">
          <Button size="sm" variant="outline" onClick={() => setImportDialogOpen(true)}>
            <Upload className="w-4 h-4 mr-1" />Import
          </Button>
          <Button size="sm" onClick={() => actions.handleCreateNew(isCreating, setIsCreating)} disabled={isCreating}>
            {isCreating ? <Loader2 className="w-4 h-4 mr-1 animate-spin" /> : <Plus className="w-4 h-4 mr-1" />}Create New
          </Button>
        </div>
      </div>

      {showSkeleton ? (
        <div className="space-y-2">
          {Array.from({ length: 4 }).map((_, i) => (<Skeleton key={`skeleton-${i}`} className="h-12 w-full rounded" />))}
        </div>
      ) : isError ? (
        <p className="text-sm text-muted-foreground py-8 text-center">Failed to load workflows.</p>
      ) : workflows.length === 0 ? (
        <p className="text-sm text-muted-foreground py-8 text-center">No workflows yet. Create one to get started.</p>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Integrations</TableHead>
              <TableHead className="w-[160px]" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {workflows.map((workflow) => (
              <WorkflowRow
                key={workflow.workflow_id}
                workflow={workflow}
                renamingId={renamingId}
                renameValue={renameValue}
                setRenameValue={setRenameValue}
                onNavigate={(id) => actions.navigate(`/workflows/${id}`)}
                onRenameSave={(id) => actions.handleRenameSave(id, renameValue, setRenamingId)}
                onRenameCancel={() => setRenamingId(null)}
                onRenameStart={(w) => { setRenamingId(w.workflow_id); setRenameValue(w.name); }}
                onExport={actions.handleExport}
                onDuplicate={actions.handleDuplicate}
                onDelete={setDeleteTarget}
              />
            ))}
          </TableBody>
        </Table>
      )}

      <DeleteWorkflowDialog
        open={!!deleteTarget}
        onOpenChange={(open) => !open && setDeleteTarget(null)}
        workflowName={deleteTarget?.name ?? ''}
        onConfirm={() => actions.handleDeleteConfirm(deleteTarget, setIsDeleting, setDeleteTarget)}
        isDeleting={isDeleting}
      />

      <WorkflowImportDialog
        open={importDialogOpen}
        onOpenChange={setImportDialogOpen}
        onImport={actions.handleImport}
      />
    </div>
  );
}
