import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';

interface SimplifyWarningDialogProps {
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Warning dialog shown when switching from Full HTML to Simple Editor mode
 * with complex HTML content that will be simplified.
 */
export function SimplifyWarningDialog({ open, onConfirm, onCancel }: SimplifyWarningDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Simplify HTML Content?</AlertDialogTitle>
          <AlertDialogDescription>
            Your HTML contains elements that will be simplified when switching to Simple Editor mode.
            Tables, inline styles, images, and other advanced formatting will be removed or converted to basic text.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel onClick={onCancel}>
            Keep Full HTML
          </AlertDialogCancel>
          <AlertDialogAction onClick={onConfirm}>
            Switch Anyway
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
