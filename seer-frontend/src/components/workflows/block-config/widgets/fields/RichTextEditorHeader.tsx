import { Code2, Eye, FileCode2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { RichTextToolbar } from './RichTextToolbar';
import type { Editor } from '@tiptap/react';
import type { ContentMode, EditorMode } from './types';

interface RichTextEditorHeaderProps {
  editor: Editor | null;
  /** Allowed features from tool schema's x-rich-text-features */
  features?: string[];
  /** Output format - 'full-html' mode only available when outputFormat is 'html' */
  outputFormat: 'html' | 'markdown';
  contentMode: ContentMode;
  editorMode: EditorMode;
  onContentModeChange: (mode: ContentMode) => void;
  onToggleSubMode: () => void;
}

/**
 * Header component for RichTextField with mode selector and formatting toolbar.
 *
 * Layout:
 * ┌────────────────────────────────────────────────────────────────────┐
 * │ [Simple Editor ▾]  │  B I U S Link ••• │           │ [</>] or [👁] │
 * └────────────────────────────────────────────────────────────────────┘
 *
 * - Mode selector: Switch between Simple Editor (Tiptap) and Full HTML
 * - Toolbar: Only visible in Simple Editor mode
 * - Sub-mode toggle: HTML/Preview toggle (varies by content mode)
 */
export function RichTextEditorHeader({
  editor,
  features,
  outputFormat,
  contentMode,
  editorMode,
  onContentModeChange,
  onToggleSubMode,
}: RichTextEditorHeaderProps) {
  const isSimpleMode = contentMode === 'simple';
  const showToolbar = isSimpleMode && editorMode === 'visual';
  const supportsFullHtml = outputFormat === 'html';

  // Determine toggle button label and icon based on current state
  const getToggleButton = () => {
    if (isSimpleMode) {
      // Simple mode: toggle between visual and html-source
      if (editorMode === 'visual') {
        return { icon: Code2, label: 'HTML', tooltip: 'Edit HTML source' };
      }
      return { icon: Eye, label: 'Preview', tooltip: 'Switch to visual editor' };
    }
    // Full HTML mode: toggle between html-source and html-preview
    if (editorMode === 'html-source') {
      return { icon: Eye, label: 'Preview', tooltip: 'Preview rendered HTML' };
    }
    return { icon: Code2, label: 'Edit HTML', tooltip: 'Edit HTML source' };
  };

  const toggleButton = getToggleButton();
  const ToggleIcon = toggleButton.icon;

  return (
    <div className="flex items-center justify-between border-b border-border bg-muted/30">
      <div className="flex items-center">
        {/* Mode selector dropdown - only show when Full HTML is available */}
        {supportsFullHtml ? (
          <Select value={contentMode} onValueChange={onContentModeChange}>
            <SelectTrigger className="w-[130px] h-7 text-xs border-0 bg-transparent focus:ring-0 focus:ring-offset-0">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="simple" className="text-xs">
                Simple Editor
              </SelectItem>
              <SelectItem value="full-html" className="text-xs">
                <span className="flex items-center gap-1">
                  <FileCode2 className="h-3 w-3" />
                  Full HTML
                </span>
              </SelectItem>
            </SelectContent>
          </Select>
        ) : (
          <div className="px-2 py-1 text-xs text-muted-foreground font-medium">
            Editor
          </div>
        )}

        {/* Separator between mode selector and toolbar */}
        {showToolbar && (
          <div className="h-5 w-px bg-border mx-1" />
        )}

        {/* Toolbar - only visible in simple mode with visual editor */}
        {showToolbar && <RichTextToolbar editor={editor} features={features} />}

        {/* Label for HTML source mode */}
        {editorMode === 'html-source' && (
          <div className="px-2 py-1 text-xs text-muted-foreground font-medium">
            HTML Source
          </div>
        )}

        {/* Label for preview mode */}
        {editorMode === 'html-preview' && (
          <div className="px-2 py-1 text-xs text-muted-foreground font-medium">
            Preview
          </div>
        )}
      </div>

      {/* Sub-mode toggle button - only show when HTML editing is relevant */}
      {supportsFullHtml && (
        <TooltipProvider delayDuration={300}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onToggleSubMode}
                className={cn(
                  'h-7 px-2 mr-1',
                  editorMode !== 'visual' && 'bg-muted text-foreground'
                )}
              >
                <ToggleIcon className="h-4 w-4 mr-1" />
                <span className="text-xs">{toggleButton.label}</span>
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="text-xs">
              {toggleButton.tooltip}
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
    </div>
  );
}
