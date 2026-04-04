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
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { type Memory } from "@/lib/memory-api";

const MAX_MEMORY_LENGTH = 2000;

interface MemoryFormDialogProps {
  open: boolean;
  mode: "create" | "edit";
  memory: Memory | null;
  onOpenChange: (open: boolean) => void;
  onSubmit: (memory: string) => void;
  isSaving?: boolean;
}

export function MemoryFormDialog({
  open,
  mode,
  memory,
  onOpenChange,
  onSubmit,
  isSaving = false,
}: MemoryFormDialogProps) {
  const [value, setValue] = useState("");

  useEffect(() => {
    if (!open) {
      setValue("");
      return;
    }

    setValue(mode === "edit" ? memory?.memory ?? "" : "");
  }, [memory, mode, open]);

  const trimmedValue = value.trim();
  const isEmpty = trimmedValue.length === 0;
  const isTooLong = value.length > MAX_MEMORY_LENGTH;
  const isSubmitDisabled = isSaving || isEmpty || isTooLong;

  const handleSubmit = () => {
    if (isSubmitDisabled) return;
    onSubmit(trimmedValue);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{mode === "edit" ? "Edit Memory" : "Add Memory"}</DialogTitle>
          <DialogDescription>
            {mode === "edit"
              ? "Update what the agent should remember."
              : "Add a memory the agent should keep for future conversations."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-2">
          <Label htmlFor="memory-content">Memory</Label>
          <Textarea
            id="memory-content"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            placeholder="Example: I prefer concise answers with code examples."
            className="min-h-[140px]"
            maxLength={MAX_MEMORY_LENGTH}
            disabled={isSaving}
          />
          <div className="flex items-center justify-between text-xs">
            <span className={isEmpty ? "text-destructive" : "text-muted-foreground"}>
              {isEmpty ? "Memory content is required." : " "}
            </span>
            <span className="text-muted-foreground">
              {value.length}/{MAX_MEMORY_LENGTH}
            </span>
          </div>
        </div>

        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button type="button" onClick={handleSubmit} disabled={isSubmitDisabled}>
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                {mode === "edit" ? "Saving..." : "Adding..."}
              </>
            ) : mode === "edit" ? (
              "Save Changes"
            ) : (
              "Add Memory"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
