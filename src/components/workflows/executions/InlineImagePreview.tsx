/**
 * InlineImagePreview Component
 *
 * Lightweight inline preview for image URLs in HITL display items.
 * Shows a constrained thumbnail that expands to full-size in a dialog.
 */
import { useState, useCallback } from 'react';
import { Loader2, ExternalLink, ImageOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { cn } from '@/lib/utils';

export interface InlineImagePreviewProps {
  src: string;
  alt?: string;
  className?: string;
}

type LoadState = 'loading' | 'loaded' | 'error';

export function InlineImagePreview({ src, alt = 'Image preview', className }: InlineImagePreviewProps) {
  const [loadState, setLoadState] = useState<LoadState>('loading');
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const handleLoad = useCallback(() => {
    setLoadState('loaded');
  }, []);

  const handleError = useCallback(() => {
    setLoadState('error');
  }, []);

  const handleOpenInNewTab = useCallback(() => {
    window.open(src, '_blank', 'noopener,noreferrer');
  }, [src]);

  // Error state
  if (loadState === 'error') {
    return (
      <div
        className={cn(
          'flex items-center gap-2 px-2 py-1.5 rounded-md',
          'bg-muted/50 text-muted-foreground text-xs',
          className
        )}
      >
        <ImageOff className="w-4 h-4 shrink-0" />
        <span className="truncate">Failed to load image</span>
      </div>
    );
  }

  return (
    <>
      {/* Thumbnail */}
      <button
        type="button"
        onClick={() => setIsDialogOpen(true)}
        className={cn(
          'relative rounded-md overflow-hidden border border-border/50',
          'hover:border-border hover:ring-2 hover:ring-ring/20 transition-all',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
          'cursor-pointer bg-muted/30',
          className
        )}
        aria-label={`View ${alt} full size`}
      >
        {/* Loading state */}
        {loadState === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center bg-muted/50">
            <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
          </div>
        )}

        {/* Image */}
        <img
          src={src}
          alt={alt}
          onLoad={handleLoad}
          onError={handleError}
          className={cn(
            'max-h-32 max-w-full object-contain',
            loadState === 'loading' && 'opacity-0'
          )}
        />
      </button>

      {/* Full-size dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between gap-4">
              <span className="truncate">{alt}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleOpenInNewTab}
                className="shrink-0"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Open in new tab
              </Button>
            </DialogTitle>
          </DialogHeader>
          <div className="flex items-center justify-center p-4">
            <img
              src={src}
              alt={alt}
              className="max-w-full max-h-[70vh] object-contain rounded-md"
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
