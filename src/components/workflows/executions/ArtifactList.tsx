import { useState } from 'react';
import { File, FileText, ImageIcon, Download, ChevronDown, Loader2, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';
import { filesApi, formatFileSize, type UserFile } from '@/lib/files-api';
import { toast } from '@/components/ui/use-toast';
import { FilePreviewDialog } from '@/components/data-storage/FilePreviewDialog';
import type { WorkflowFileRef } from './types';

function fileIcon(mimeType: string) {
  if (mimeType === 'application/pdf') return FileText;
  if (mimeType.startsWith('image/')) return ImageIcon;
  return File;
}

/** Convert WorkflowFileRef to UserFile shape required by FilePreviewDialog. */
function toUserFile(artifact: WorkflowFileRef): UserFile {
  return {
    file_id: artifact.file_id,
    filename: artifact.filename,
    mime_type: artifact.mime_type,
    size_bytes: artifact.size_bytes,
    size_human: formatFileSize(artifact.size_bytes),
    run_id: artifact.workflow_run_id,
    workflow_id: null,
    workflow_name: null,
    source_node_id: null,
    source_tool: null,
    created_at: artifact.created_at,
  };
}

interface ArtifactRowProps {
  artifact: WorkflowFileRef;
  onPreview: (artifact: WorkflowFileRef) => void;
}

function ArtifactRow({ artifact, onPreview }: ArtifactRowProps) {
  const [loading, setLoading] = useState(false);
  const Icon = fileIcon(artifact.mime_type);

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setLoading(true);
    try {
      const { download_url } = await filesApi.getDownloadUrl(artifact.file_id);
      window.open(download_url, '_blank');
    } catch {
      toast({ title: 'Download failed', description: `Could not download ${artifact.filename}.`, variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-2 py-1.5 px-2 rounded-md hover:bg-muted/50 transition-colors group">
      <Icon className="h-4 w-4 text-muted-foreground shrink-0" />
      <span className="text-xs truncate flex-1 min-w-0">{artifact.filename}</span>
      <span className="text-xs text-muted-foreground shrink-0">{formatFileSize(artifact.size_bytes)}</span>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={() => onPreview(artifact)}
        title="Preview"
      >
        <Eye className="h-3 w-3" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        className="h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={handleDownload}
        disabled={loading}
        title="Download"
      >
        {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <Download className="h-3 w-3" />}
      </Button>
    </div>
  );
}

interface ArtifactListProps {
  artifacts?: WorkflowFileRef[];
}

export function ArtifactList({ artifacts }: ArtifactListProps) {
  const [isOpen, setIsOpen] = useState(true);
  const [previewArtifact, setPreviewArtifact] = useState<WorkflowFileRef | null>(null);

  if (!artifacts || artifacts.length === 0) return null;

  return (
    <>
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <div className="rounded-lg border bg-card">
          <CollapsibleTrigger asChild>
            <button className="flex items-center justify-between w-full p-3 text-left hover:bg-muted/50 transition-colors">
              <div className="flex items-center gap-2">
                <ChevronDown className={cn('h-4 w-4 transition-transform', !isOpen && '-rotate-90')} />
                <span className="text-sm font-medium">Artifacts</span>
                <span className="text-xs text-muted-foreground">({artifacts.length})</span>
              </div>
            </button>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-2 pb-2">
              {artifacts.map((artifact) => (
                <ArtifactRow
                  key={artifact.file_id}
                  artifact={artifact}
                  onPreview={setPreviewArtifact}
                />
              ))}
            </div>
          </CollapsibleContent>
        </div>
      </Collapsible>

      <FilePreviewDialog
        open={previewArtifact !== null}
        onOpenChange={(open) => { if (!open) setPreviewArtifact(null); }}
        file={previewArtifact ? toUserFile(previewArtifact) : null}
      />
    </>
  );
}
