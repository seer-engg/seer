import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';

export interface ResourceDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  trigger: React.ReactNode;
  children: React.ReactNode;
}

export function ResourceDialog({
  open,
  onOpenChange,
  title = 'Browse Resources',
  trigger,
  children,
}: ResourceDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {trigger}
      <DialogContent className="sm:max-w-[500px] max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
        </DialogHeader>
        {children}
      </DialogContent>
    </Dialog>
  );
}
