import { useCallback, useState } from 'react';
import { Clock, GitBranch, Loader2, Play, Share2, Tag } from 'lucide-react';

import { Button } from '@/components/ui/button';
import type { WorkflowLifecycleStatus } from '@/components/workflows/buildtypes';
import type { WorkflowVersionListItem } from '@/types/workflow-spec';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { useWorkflowStore } from '@/stores/workflowStore';
import { useUIStore } from '@/stores/uiStore';
import { ShareAsTemplateDialog } from './ShareAsTemplateDialog';
import { useWorkflowDetailQuery } from '@/hooks/useWorkflowQueries';

interface WorkflowLifecycleBarProps {
  lifecycleStatus?: WorkflowLifecycleStatus | null;
  onRunClick: () => void;
  onPublishClick: () => void;
  isExecuting?: boolean;
  isPublishing?: boolean;
  canRun?: boolean;
  canPublish?: boolean;
  publishDisabledReason?: string;
  runDisabledReason?: string;
  versionOptions?: WorkflowVersionListItem[];
  onVersionRestore?: (versionId: number) => void;
  isVersionsLoading?: boolean;
  isRestoringVersion?: boolean;
  versionRestoreDisabledReason?: string;
  workflowId?: string | null;
}

const formatHistoryLabel = (version: WorkflowVersionListItem): string => {
  const versionNumber = version.version_number ?? version.version_id;
  const status = version.status?.toLowerCase?.();
  const qualifiers = [
    status,
    version.is_published ? 'published' : null,
    version.is_latest ? 'latest' : null,
  ].filter(Boolean);
  return `v${versionNumber}${qualifiers.length ? ` · ${qualifiers.join(' · ')}` : ''}`;
};

function VersionsDropdownContent({
  lifecycleStatus,
  isVersionsLoading,
  versionOptions,
  onVersionRestore,
}: {
  lifecycleStatus?: WorkflowLifecycleStatus | null;
  isVersionsLoading?: boolean;
  versionOptions?: WorkflowVersionListItem[];
  onVersionRestore?: (versionId: number) => void;
}) {
  const hasVersionOptions = Boolean(versionOptions && versionOptions.length > 0);

  return (
    <DropdownMenuContent align="end" className="w-56">
      {isVersionsLoading && (
        <DropdownMenuItem disabled>
          <Loader2 className="mr-2 h-3 w-3 animate-spin" />
          Loading versions…
        </DropdownMenuItem>
      )}
      {!isVersionsLoading && hasVersionOptions ? (
        versionOptions!.map((version) => (
          <DropdownMenuItem
            key={version.version_id}
            onSelect={(event) => {
              event.preventDefault();
              onVersionRestore?.(version.version_id);
            }}
          >
            <div className="flex flex-col gap-0.5 text-xs">
              <span className="font-medium text-foreground">{formatHistoryLabel(version)}</span>
            </div>
          </DropdownMenuItem>
        ))
      ) : (
        !isVersionsLoading && <DropdownMenuItem disabled>No versions available yet</DropdownMenuItem>
      )}
    </DropdownMenuContent>
  );
}

function VersionsButton({
  lifecycleStatus,
  versionOptions,
  onVersionRestore,
  isVersionsLoading,
  isRestoringVersion,
  versionRestoreDisabledReason,
}: Pick<
  WorkflowLifecycleBarProps,
  | 'lifecycleStatus'
  | 'versionOptions'
  | 'onVersionRestore'
  | 'isVersionsLoading'
  | 'isRestoringVersion'
  | 'versionRestoreDisabledReason'
>) {
  const hasVersionOptions = Boolean(versionOptions && versionOptions.length > 0);
  const isDisabled =
    !hasVersionOptions ||
    isVersionsLoading ||
    isRestoringVersion ||
    Boolean(versionRestoreDisabledReason);

  return (
    <Tooltip>
      <DropdownMenu>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              size="sm"
              variant="ghost"
              className="h-8 px-3 gap-2"
              disabled={isDisabled}
              aria-label="Version History"
            >
              {isRestoringVersion ? <Loader2 className="h-4 w-4 animate-spin" /> : <GitBranch className="h-4 w-4" />}
              <span className="text-xs font-medium">Versions</span>
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={8}>
          <div className="flex flex-col gap-1">
            <p className="font-medium text-sm">Version History</p>
            {isDisabled && (versionRestoreDisabledReason || !hasVersionOptions) && (
              <p className="text-xs text-muted-foreground">
                {versionRestoreDisabledReason ?? 'Run tests to create checkpoint versions'}
              </p>
            )}
          </div>
        </TooltipContent>
        <VersionsDropdownContent
          lifecycleStatus={lifecycleStatus}
          isVersionsLoading={isVersionsLoading}
          versionOptions={versionOptions}
          onVersionRestore={onVersionRestore}
        />
      </DropdownMenu>
    </Tooltip>
  );
}

function RunButton({
  onRunClick,
  isExecuting,
  canRun,
  runDisabledReason,
}: Pick<WorkflowLifecycleBarProps, 'onRunClick' | 'isExecuting' | 'canRun' | 'runDisabledReason'>) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={onRunClick}
          disabled={!canRun || isExecuting}
          size="sm"
          variant="outline"
          className="h-8 px-4 gap-2"
          aria-label="Test Workflow"
          data-testid="run-button"
        >
          {isExecuting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          <span className="text-xs font-medium">Run</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={8}>
        <div className="flex flex-col gap-1">
          <p className="font-medium text-sm">{isExecuting ? 'Testing Workflow' : 'Test Workflow'}</p>
          {(!canRun && runDisabledReason) || !isExecuting ? (
            <p className="text-xs text-muted-foreground">{runDisabledReason ?? 'Run a test execution'}</p>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function PublishButton({
  onPublishClick,
  isPublishing,
  canPublish,
  publishDisabledReason,
}: Pick<WorkflowLifecycleBarProps, 'onPublishClick' | 'isPublishing' | 'canPublish' | 'publishDisabledReason'>) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={onPublishClick}
          disabled={!canPublish || isPublishing}
          size="sm"
          variant="brand"
          className="h-8 px-4 gap-2"
          aria-label="Activate"
          data-testid="publish-button"
        >
          {isPublishing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Tag className="h-4 w-4" />}
          <span className="text-xs font-medium">Activate</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={8}>
        <div className="flex flex-col gap-1">
          <p className="font-medium text-sm">{isPublishing ? 'Activating...' : 'Activate'}</p>
          {(!canPublish && publishDisabledReason) || !isPublishing ? (
            <p className="text-xs text-muted-foreground">{publishDisabledReason ?? 'Make this version live'}</p>
          ) : null}
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function ExecutionsButton({ workflowId }: { workflowId?: string | null }) {
  const rightPaneMode = useUIStore((state) => state.rightPaneMode);
  const setRightPaneMode = useUIStore((state) => state.setRightPaneMode);

  const handleClick = useCallback(() => {
    setRightPaneMode(rightPaneMode === 'executions' ? null : 'executions');
  }, [rightPaneMode, setRightPaneMode]);

  const isActive = rightPaneMode === 'executions';

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          onClick={handleClick}
          disabled={!workflowId}
          size="sm"
          variant={isActive ? 'secondary' : 'ghost'}
          className="h-8 px-3 gap-2"
          aria-label="Runs"
          data-testid="executions-button"
        >
          <Clock className={`h-4 w-4 ${isActive ? 'text-primary' : ''}`} />
          <span className="text-xs font-medium">Runs</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={8}>
        <div className="flex flex-col gap-1">
          <p className="font-medium text-sm">Run History</p>
          <p className="text-xs text-muted-foreground">
            {workflowId ? 'View past runs' : 'Save the workflow first'}
          </p>
        </div>
      </TooltipContent>
    </Tooltip>
  );
}

function ShareButton({ workflowId }: { workflowId?: string | null }) {
  const [open, setOpen] = useState(false);
  const workflowName = useWorkflowStore((state) => state.workflowName) || 'My Workflow';
  const { data: workflow } = useWorkflowDetailQuery(workflowId);
  const isPublished = workflow?.is_published ?? false;

  const label = isPublished ? 'Update Template' : 'Publish';
  const tooltip = isPublished ? 'Update your published template' : 'Publish as a reusable template';

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            onClick={() => setOpen(true)}
            disabled={!workflowId}
            size="sm"
            variant="ghost"
            className="h-8 px-3 gap-2"
            aria-label={label}
          >
            <Share2 className="h-4 w-4" />
            <span className="text-xs font-medium">{label}</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={8}>
          <div className="flex flex-col gap-1">
            <p className="font-medium text-sm">{label}</p>
            <p className="text-xs text-muted-foreground">{tooltip}</p>
          </div>
        </TooltipContent>
      </Tooltip>
      {workflowId && (
        <ShareAsTemplateDialog
          open={open}
          onOpenChange={setOpen}
          workflowId={workflowId}
          workflowName={workflowName}
        />
      )}
    </>
  );
}

function BarSeparator() {
  return <div className="w-px h-4 bg-border mx-1 shrink-0" />;
}

export function WorkflowLifecycleBar(props: WorkflowLifecycleBarProps) {
  return (
    <div className="bg-card/95 backdrop-blur border rounded-full shadow-xl px-1 py-1 flex items-center gap-0.5">
      <VersionsButton
        lifecycleStatus={props.lifecycleStatus}
        versionOptions={props.versionOptions}
        onVersionRestore={props.onVersionRestore}
        isVersionsLoading={props.isVersionsLoading}
        isRestoringVersion={props.isRestoringVersion}
        versionRestoreDisabledReason={props.versionRestoreDisabledReason}
      />
      <BarSeparator />
      <RunButton
        onRunClick={props.onRunClick}
        isExecuting={props.isExecuting}
        canRun={props.canRun}
        runDisabledReason={props.runDisabledReason}
      />
      <PublishButton
        onPublishClick={props.onPublishClick}
        isPublishing={props.isPublishing}
        canPublish={props.canPublish}
        publishDisabledReason={props.publishDisabledReason}
      />
      <ShareButton workflowId={props.workflowId} />
      <BarSeparator />
      <ExecutionsButton workflowId={props.workflowId} />
    </div>
  );
}
