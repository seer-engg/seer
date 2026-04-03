import { useState } from "react";
import { Loader2, AlertTriangle } from "lucide-react";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface DeleteAllMemoriesDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  totalCount: number;
  onConfirm: () => void;
  isDeleting?: boolean;
}

const CONFIRMATION_TEXT = "DELETE ALL";

export function DeleteAllMemoriesDialog({
  open,
  onOpenChange,
  totalCount,
  onConfirm,
  isDeleting = false,
}: DeleteAllMemoriesDialogProps) {
  const [confirmInput, setConfirmInput] = useState("");
  const isConfirmValid = confirmInput === CONFIRMATION_TEXT;

  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) setConfirmInput("");
    onOpenChange(newOpen);
  };

  const handleConfirm = () => {
    if (isConfirmValid) {
      onConfirm();
    }
  };

  return (
    <AlertDialog open={open} onOpenChange={handleOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle className="flex items-center gap-2 text-destructive">
            <AlertTriangle className="h-5 w-5" />
            Delete All Memories
          </AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-3">
              <span className="block">
                This will permanently delete{" "}
                <strong>{totalCount}</strong>{" "}
                {totalCount === 1 ? "memory" : "memories"}. This action cannot be
                undone and is intended for GDPR compliance requests.
              </span>
              <div className="space-y-2 pt-2">
                <Label htmlFor="confirm-delete" className="text-foreground">
                  Type <strong>{CONFIRMATION_TEXT}</strong> to confirm:
                </Label>
                <Input
                  id="confirm-delete"
                  value={confirmInput}
                  onChange={(e) => setConfirmInput(e.target.value)}
                  placeholder={CONFIRMATION_TEXT}
                  disabled={isDeleting}
                  autoComplete="off"
                />
              </div>
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={handleConfirm}
            disabled={isDeleting || !isConfirmValid}
            className={cn(buttonVariants({ variant: "destructive" }))}
          >
            {isDeleting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Deleting All...
              </>
            ) : (
              "Delete All Memories"
            )}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
