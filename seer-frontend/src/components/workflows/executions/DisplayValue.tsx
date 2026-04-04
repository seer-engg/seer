/**
 * DisplayValue Component
 *
 * Auto-detecting display value renderer for HITL display items.
 * Routes to appropriate renderer based on value type:
 * - Image URLs -> InlineImagePreview
 * - File references -> FileRefImagePreview
 * - Other values -> Text display
 */
import { detectImageDisplay } from '@/lib/image-detection';
import { InlineImagePreview } from './InlineImagePreview';
import { FileRefImagePreview } from './FileRefImagePreview';
import { cn } from '@/lib/utils';

export interface DisplayValueProps {
  value: unknown;
  className?: string;
}

/**
 * Format a value for text display
 */
function formatTextValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '-';
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

export function DisplayValue({ value, className }: DisplayValueProps) {
  const imageResult = detectImageDisplay(value);

  // Image URL - render inline preview
  if (imageResult.type === 'url') {
    return (
      <InlineImagePreview
        src={imageResult.url}
        alt="Display image"
        className={className}
      />
    );
  }

  // File reference - fetch and render
  if (imageResult.type === 'file_ref') {
    return (
      <FileRefImagePreview
        fileRef={imageResult.fileRef}
        className={className}
      />
    );
  }

  // Default: text display (no truncation - use break-words for wrapping)
  return (
    <span className={cn('text-sm font-medium break-words whitespace-pre-wrap', className)}>
      {formatTextValue(value)}
    </span>
  );
}
