/**
 * Toolbar for browser viewer with URL bar and controls.
 */
import { useState, useCallback } from 'react';
import { ArrowLeft, ArrowRight, RotateCw, Home, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { BrowserViewerStatus } from './BrowserViewerStatus';
import type { StreamStatus } from '@/types/browser';

interface BrowserViewerToolbarProps {
  currentUrl: string;
  status: StreamStatus;
  error?: string | null;
  onNavigate: (url: string) => void;
  onRefresh?: () => void;
  disabled?: boolean;
}

export function BrowserViewerToolbar({
  currentUrl,
  status,
  error,
  onNavigate,
  onRefresh,
  disabled = false,
}: BrowserViewerToolbarProps) {
  const [urlInput, setUrlInput] = useState(currentUrl);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!urlInput.trim()) return;

      // Add https:// if no protocol
      let url = urlInput.trim();
      if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = `https://${url}`;
      }
      onNavigate(url);
    },
    [urlInput, onNavigate]
  );

  const handleRefresh = useCallback(() => {
    if (currentUrl) {
      onNavigate(currentUrl);
    }
    onRefresh?.();
  }, [currentUrl, onNavigate, onRefresh]);

  const handleHome = useCallback(() => {
    onNavigate('about:blank');
  }, [onNavigate]);

  return (
    <div className="flex items-center gap-2 p-2 bg-muted/50 border-b rounded-t-lg">
      <div className="flex items-center gap-0.5">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={disabled}
          onClick={() => {/* Browser history not supported via CDP */}}
          title="Back (not supported)"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={disabled}
          onClick={() => {/* Browser history not supported via CDP */}}
          title="Forward (not supported)"
        >
          <ArrowRight className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={disabled || !currentUrl}
          onClick={handleRefresh}
          title="Refresh"
        >
          <RotateCw className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={disabled}
          onClick={handleHome}
          title="Home"
        >
          <Home className="h-3.5 w-3.5" />
        </Button>
      </div>

      <form onSubmit={handleSubmit} className="flex-1 flex items-center gap-2">
        <Input
          value={urlInput}
          onChange={(e) => setUrlInput(e.target.value)}
          placeholder="Enter URL..."
          disabled={disabled}
          className="h-7 text-xs"
        />
        <Button
          type="submit"
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          disabled={disabled || !urlInput.trim()}
          title="Go"
        >
          <ExternalLink className="h-3.5 w-3.5" />
        </Button>
      </form>

      <BrowserViewerStatus status={status} error={error} />
    </div>
  );
}
