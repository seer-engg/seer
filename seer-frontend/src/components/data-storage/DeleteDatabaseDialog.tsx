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
import type { IntegrationResource } from '@/lib/api-client';

interface DeleteDatabaseDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  resource: IntegrationResource | null;
  onConfirm: () => void;
  isPending: boolean;
}

export function DeleteDatabaseDialog({
  open,
  onOpenChange,
  resource,
  onConfirm,
  isPending,
}: DeleteDatabaseDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Remove Database Connection</AlertDialogTitle>
          <AlertDialogDescription>
            Are you sure you want to remove{' '}
            <span className="font-medium">{resource?.name || 'this database'}</span>?
            This will revoke access and any workflows using this database will need to be
            reconfigured.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
          >
            {isPending ? 'Removing...' : 'Remove'}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
