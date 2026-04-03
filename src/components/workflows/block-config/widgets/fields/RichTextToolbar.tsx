import { type Editor } from '@tiptap/react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { createToolbarFeatures } from './toolbar-features';

interface ToolbarButtonProps {
  onClick: () => void;
  isActive?: boolean;
  disabled?: boolean;
  tooltip: string;
  children: React.ReactNode;
}

function ToolbarButton({
  onClick,
  isActive,
  disabled,
  tooltip,
  children,
}: ToolbarButtonProps) {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClick}
            disabled={disabled}
            className={cn(
              'h-7 w-7 p-0',
              isActive && 'bg-muted text-foreground'
            )}
          >
            {children}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="text-xs">
          {tooltip}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

interface RichTextToolbarProps {
  editor: Editor | null;
  /** Allowed features from tool schema's x-rich-text-features. If undefined, all features shown. */
  features?: string[];
}

export function RichTextToolbar({ editor, features }: RichTextToolbarProps) {
  if (!editor) return null;

  const allFeatures = createToolbarFeatures(editor);
  const supportedFeatures = features
    ? allFeatures.filter((f) => features.includes(f.feature))
    : allFeatures;

  return (
    <div className="flex flex-wrap items-center gap-0.5 border-b border-border bg-muted/30 px-1 py-1">
      {supportedFeatures.map((f) => (
        <ToolbarButton
          key={f.feature}
          onClick={f.action}
          isActive={f.isActive()}
          tooltip={f.tooltip}
        >
          {f.icon}
        </ToolbarButton>
      ))}
    </div>
  );
}
