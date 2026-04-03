import { ArrowLeft, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

interface ExecutionTraceHeaderProps {
  runId: string;
  workflowId?: string | null;
  workflowName?: string | null;
  onBack: () => void;
}

export function ExecutionTraceHeader({
  runId,
  workflowId,
  workflowName,
  onBack,
}: ExecutionTraceHeaderProps) {
  return (
    <header className="h-14 border-b border-border bg-card shrink-0">
      <div className="max-w-7xl mx-auto h-full px-6 flex items-center gap-4">
        {/* Back Button */}
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </Button>

        {/* Separator */}
        <Separator orientation="vertical" className="h-6" />

        {/* Breadcrumb Trail */}
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Link to="/workflows" className="hover:text-foreground transition-colors">
            Workflows
          </Link>
          <ChevronRight className="w-4 h-4" />
          {workflowName ? (
            <Link
              to={`/workflows/${workflowId}`}
              className="hover:text-foreground transition-colors"
            >
              {workflowName}
            </Link>
          ) : (
            <span>Workflow</span>
          )}
          <ChevronRight className="w-4 h-4" />
          <span className="font-medium text-foreground">
            {runId}
          </span>
        </div>
      </div>
    </header>
  );
}
