/**
 * FileRefImagePreview Component
 *
 * Handles workflow_file_ref objects by fetching presigned URLs
 * and rendering them with InlineImagePreview.
 */
import { useState, useEffect } from 'react';
import { Loader2, ImageOff, Lock } from 'lucide-react';
import { filesApi } from '@/lib/files-api';
import { InlineImagePreview } from './InlineImagePreview';
import type { WorkflowFileRef } from '@/lib/image-detection';
import { cn } from '@/lib/utils';

export interface FileRefImagePreviewProps {
  fileRef: WorkflowFileRef;
  className?: string;
}

type FetchState =
  | { status: 'loading' }
  | { status: 'success'; url: string }
  | { status: 'error'; message: string }
  | { status: 'unauthorized' };

export function FileRefImagePreview({ fileRef, className }: FileRefImagePreviewProps) {
  const [state, setState] = useState<FetchState>({ status: 'loading' });

  useEffect(() => {
    let cancelled = false;

    async function fetchUrl() {
      try {
        const response = await filesApi.getDownloadUrl(fileRef.file_id, true);
        if (!cancelled) {
          setState({ status: 'success', url: response.download_url });
        }
      } catch (error) {
        if (cancelled) return;

        // Check for auth errors (public forms can't access files API)
        if (error instanceof Error && error.message.includes('401')) {
          setState({ status: 'unauthorized' });
        } else {
          setState({
            status: 'error',
            message: error instanceof Error ? error.message : 'Failed to load image',
          });
        }
      }
    }

    fetchUrl();

    return () => {
      cancelled = true;
    };
  }, [fileRef.file_id]);

  // Loading state
  if (state.status === 'loading') {
    return (
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded-md',
          'bg-muted/50 text-muted-foreground text-xs',
          className
        )}
      >
        <Loader2 className="w-4 h-4 animate-spin shrink-0" />
        <span className="truncate">{fileRef.filename}</span>
      </div>
    );
  }

  // Unauthorized state (public form context)
  if (state.status === 'unauthorized') {
    return (
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded-md',
          'bg-muted/50 text-muted-foreground text-xs',
          className
        )}
      >
        <Lock className="w-4 h-4 shrink-0" />
        <span className="truncate">{fileRef.filename}</span>
      </div>
    );
  }

  // Error state
  if (state.status === 'error') {
    return (
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded-md',
          'bg-muted/50 text-muted-foreground text-xs',
          className
        )}
      >
        <ImageOff className="w-4 h-4 shrink-0" />
        <span className="truncate">{fileRef.filename}</span>
      </div>
    );
  }

  // Success - render InlineImagePreview
  return (
    <InlineImagePreview
      src={state.url}
      alt={fileRef.filename}
      className={className}
    />
  );
}
