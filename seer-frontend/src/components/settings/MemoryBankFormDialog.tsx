import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { MemoryBank } from "@/lib/memory-api";

interface MemoryBankFormDialogProps {
  open: boolean;
  mode: "create" | "edit";
  bank: MemoryBank | null;
  onOpenChange: (open: boolean) => void;
  onSubmit: (values: { name: string; description?: string | null }) => void;
  isSaving?: boolean;
}

export function MemoryBankFormDialog({
  open,
  mode,
  bank,
  onOpenChange,
  onSubmit,
  isSaving = false,
}: MemoryBankFormDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  useEffect(() => {
    if (!open) {
      setName("");
      setDescription("");
      return;
    }

    setName(mode === "edit" ? bank?.name ?? "" : "");
    setDescription(mode === "edit" ? bank?.description ?? "" : "");
  }, [bank, mode, open]);

  const trimmedName = name.trim();
  const trimmedDescription = description.trim();
  const isSubmitDisabled = isSaving || trimmedName.length === 0;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{mode === "edit" ? "Edit Memory Bank" : "Create Memory Bank"}</DialogTitle>
          <DialogDescription>
            {mode === "edit"
              ? "Update the bank name or description."
              : "Create a workspace memory bank for agents and memory tools."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="memory-bank-name">Name</Label>
            <Input
              id="memory-bank-name"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Customer context"
              disabled={isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="memory-bank-description">Description</Label>
            <Textarea
              id="memory-bank-description"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="What this bank should store and who uses it."
              className="min-h-[120px]"
              disabled={isSaving}
            />
          </div>
        </div>

        <DialogFooter>
          <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={isSaving}>
            Cancel
          </Button>
          <Button
            type="button"
            disabled={isSubmitDisabled}
            onClick={() =>
              onSubmit({
                name: trimmedName,
                description: trimmedDescription || null,
              })
            }
          >
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {mode === "edit" ? "Saving..." : "Creating..."}
              </>
            ) : mode === "edit" ? (
              "Save Changes"
            ) : (
              "Create Bank"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
