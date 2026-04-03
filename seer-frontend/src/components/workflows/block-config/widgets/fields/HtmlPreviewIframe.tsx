import { useMemo } from 'react';

interface HtmlPreviewIframeProps {
  html: string;
  rows?: number;
}

/**
 * Sandboxed iframe for rendering raw HTML previews.
 *
 * Security: The sandbox attribute provides isolation:
 * - No JavaScript execution (no allow-scripts)
 * - No form submissions
 * - No popups/new windows
 * - No plugin content
 *
 * allow-same-origin is needed for srcDoc to render properly,
 * but without allow-scripts, no code can actually execute.
 */
export function HtmlPreviewIframe({ html, rows = 4 }: HtmlPreviewIframeProps) {
  // For large HTML (>100KB), use Blob URL instead of srcDoc for better performance
  const { src, srcDoc } = useMemo(() => {
    const htmlSize = new Blob([html]).size;
    if (htmlSize > 100 * 1024) {
      // Large HTML: use Blob URL
      const blob = new Blob([html], { type: 'text/html' });
      return { src: URL.createObjectURL(blob), srcDoc: undefined };
    }
    // Normal HTML: use srcDoc
    return { src: undefined, srcDoc: html };
  }, [html]);

  // Calculate min height - use larger size for preview to show email templates properly
  const minHeight = Math.max(rows * 32 + 80, 800);

  return (
    <div className="relative bg-white dark:bg-zinc-900 rounded-b-md overflow-hidden border-t border-border">
      <iframe
        src={src}
        srcDoc={srcDoc}
        sandbox="allow-same-origin"
        title="HTML Preview"
        className="w-full border-0"
        style={{ minHeight: `${minHeight}px` }}
      />
      <div className="absolute bottom-2 right-2 text-xs text-muted-foreground bg-background/80 px-2 py-1 rounded">
        Preview Mode
      </div>
    </div>
  );
}
