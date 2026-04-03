import { Loader2 } from "lucide-react";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { MemoryBank } from "@/lib/memory-api";

interface DeleteMemoryBankDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bank: MemoryBank | null;
  onConfirm: () => void;
  isDeleting?: boolean;
}

export function DeleteMemoryBankDialog({
  open,
  onOpenChange,
  bank,
  onConfirm,
  isDeleting = false,
}: DeleteMemoryBankDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete Memory Bank</AlertDialogTitle>
          <AlertDialogDescription>
            {bank?.is_default
              ? "Default memory banks cannot be deleted."
              : `Delete "${bank?.name ?? "this bank"}"? This action cannot be undone.`}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={onConfirm}
            disabled={isDeleting || bank?.is_default === true}
            className={cn(buttonVariants({ variant: "destructive" }))}
          >
            {isDeleting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Deleting...
              </>
            ) : (
              "Delete Bank"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
