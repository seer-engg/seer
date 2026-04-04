/**
 * RichDisplayValue Component
 *
 * Smart display value renderer for HITL display items that handles:
 * - Markdown content with proper rendering
 * - Long content with expand/collapse functionality
 * - Images via existing preview components
 */
import { useState, useRef, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { detectImageDisplay } from '@/lib/image-detection';
import { analyzeContent, type ContentAnalysis } from '@/lib/content-analysis';
import { MarkdownContent } from '@/components/workflows/chat/MarkdownContent';
import { InlineImagePreview } from './InlineImagePreview';
import { FileRefImagePreview } from './FileRefImagePreview';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RichDisplayValueProps {
  value: unknown;
  className?: string;
  /** Pre-computed content analysis (optional, will compute if not provided) */
  analysis?: ContentAnalysis;
  /** Default collapsed height in pixels */
  collapsedHeight?: number;
}

/**
 * Collapsible wrapper for long markdown content
 */
function CollapsibleMarkdown({
  content,
  collapsedHeight = 200,
}: {
  content: string;
  collapsedHeight?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [needsCollapse, setNeedsCollapse] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (contentRef.current) {
      setNeedsCollapse(contentRef.current.scrollHeight > collapsedHeight);
    }
  }, [content, collapsedHeight]);

  return (
    <div className="relative">
      <div
        ref={contentRef}
        className={cn(
          'rounded-md border border-border bg-muted/30 p-4 overflow-hidden transition-all duration-200',
          !isExpanded && needsCollapse && 'overflow-hidden'
        )}
        style={
          !isExpanded && needsCollapse ? { maxHeight: collapsedHeight } : undefined
        }
      >
        <MarkdownContent content={content} />
      </div>

      {/* Gradient fade overlay when collapsed */}
      {!isExpanded && needsCollapse && (
        <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-muted/80 to-transparent pointer-events-none rounded-b-md" />
      )}

      {/* Expand/collapse button */}
      {needsCollapse && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-2 text-xs text-muted-foreground hover:text-foreground"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="h-3 w-3 mr-1" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="h-3 w-3 mr-1" />
              Show more
            </>
          )}
        </Button>
      )}
    </div>
  );
}

export function RichDisplayValue({
  value,
  className,
  analysis: providedAnalysis,
  collapsedHeight = 200,
}: RichDisplayValueProps) {
  // Check for image content first
  const imageResult = detectImageDisplay(value);

  if (imageResult.type === 'url') {
    return (
      <InlineImagePreview
        src={imageResult.url}
        alt="Display image"
        className={className}
      />
    );
  }

  if (imageResult.type === 'file_ref') {
    return (
      <FileRefImagePreview fileRef={imageResult.fileRef} className={className} />
    );
  }

  // Analyze content if not provided
  const analysis = providedAnalysis ?? analyzeContent(value);

  // For markdown or long content that needs collapse
  if (analysis.isMarkdown || analysis.needsCollapse) {
    return (
      <div className={className}>
        <CollapsibleMarkdown
          content={analysis.text}
          collapsedHeight={collapsedHeight}
        />
      </div>
    );
  }

  // For long content without markdown - still render with markdown component
  // (handles proper line breaks and formatting)
  if (analysis.isLong) {
    return (
      <div
        className={cn(
          'rounded-md border border-border bg-muted/30 p-4',
          className
        )}
      >
        <MarkdownContent content={analysis.text} />
      </div>
    );
  }

  // Short plain text - simple span
  return (
    <span className={cn('text-sm font-medium break-words', className)}>
      {analysis.text}
    </span>
  );
}
