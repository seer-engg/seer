import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Kbd } from '@/components/ui/kbd';
import { useShortcuts } from '@/hooks/utility/useKeyboardShortcuts';

interface KeymapDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function KeymapDialog({ open, onOpenChange }: KeymapDialogProps) {
  const shortcuts = useShortcuts();
  const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;

  // Group shortcuts by category
  const grouped = shortcuts.reduce(
    (acc, shortcut) => {
      if (!acc[shortcut.category]) {
        acc[shortcut.category] = [];
      }
      acc[shortcut.category].push(shortcut);
      return acc;
    },
    {} as Record<string, typeof shortcuts>
  );

  // Define category order
  const categoryOrder = [
    'Navigation',
    'Dialog Actions',
    'Chat & Input',
    'Canvas',
    'Help',
  ];

  // Sort categories by predefined order
  const sortedCategories = Object.keys(grouped).sort((a, b) => {
    const indexA = categoryOrder.indexOf(a);
    const indexB = categoryOrder.indexOf(b);
    if (indexA === -1 && indexB === -1) return a.localeCompare(b);
    if (indexA === -1) return 1;
    if (indexB === -1) return -1;
    return indexA - indexB;
  });

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Available keyboard shortcuts for faster navigation
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 max-h-[60vh] overflow-y-auto pr-2">
          {sortedCategories.map((category) => (
            <div key={category}>
              <h3 className="font-semibold text-sm mb-3 text-foreground">
                {category}
              </h3>
              <div className="space-y-2">
                {grouped[category].map((shortcut, idx) => (
                  <div
                    key={idx}
                    className="flex justify-between items-center py-2 px-3 rounded hover:bg-muted/50 transition-colors"
                  >
                    <span className="text-sm text-muted-foreground">
                      {shortcut.description}
                    </span>
                    <div className="flex items-center gap-1">
                      {(shortcut.modifiers?.ctrl || shortcut.modifiers?.meta) && (
                        <Kbd>{isMac ? '⌘' : 'Ctrl'}</Kbd>
                      )}
                      {shortcut.modifiers?.shift && <Kbd>⇧</Kbd>}
                      {shortcut.modifiers?.alt && <Kbd>⌥</Kbd>}
                      <Kbd>{shortcut.key.toUpperCase()}</Kbd>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {shortcuts.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-8">
              No keyboard shortcuts registered yet.
            </p>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
