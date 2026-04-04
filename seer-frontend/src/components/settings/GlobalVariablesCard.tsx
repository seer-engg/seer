import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { useVariablesStore } from "@/stores/variablesStore";
import { useToast } from "@/hooks/utility/use-toast";
import { Loader2, Pencil, Plus, Trash2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

function AddEditDialog({
  open,
  onOpenChange,
  editingVar,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  editingVar: { id: number; key: string; value: string; is_secret: boolean; description: string | null } | null;
}) {
  const { addVariable, editVariable } = useVariablesStore();
  const { toast } = useToast();
  const [key, setKey] = useState("");
  const [value, setValue] = useState("");
  const [isSecret, setIsSecret] = useState(false);
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (editingVar) {
      setKey(editingVar.key);
      setValue(editingVar.value);
      setIsSecret(editingVar.is_secret);
      setDescription(editingVar.description || "");
    } else {
      setKey("");
      setValue("");
      setIsSecret(false);
      setDescription("");
    }
  }, [editingVar, open]);

  const handleSave = async () => {
    if (!key.trim()) return;
    setSaving(true);
    try {
      if (editingVar) {
        await editVariable(editingVar.id, { value, is_secret: isSecret, description: description || undefined });
      } else {
        await addVariable({ key: key.trim(), value, is_secret: isSecret, description: description || undefined });
      }
      onOpenChange(false);
    } catch (e) {
      toast({ title: "Error", description: e instanceof Error ? e.message : "Failed to save variable", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{editingVar ? "Edit Variable" : "Add Variable"}</DialogTitle>
          <DialogDescription>
            Variables are accessible in workflows via <code className="text-xs bg-muted px-1 rounded">{"{{vars.key}}"}</code>
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <Label>Key</Label>
            <Input value={key} onChange={(e) => setKey(e.target.value)} disabled={!!editingVar} placeholder="e.g. api_base_url" />
          </div>
          <div>
            <Label>Value</Label>
            <Input
              value={value}
              onChange={(e) => setValue(e.target.value)}
              type={isSecret ? "password" : "text"}
              placeholder="Variable value"
            />
          </div>
          <div>
            <Label>Description (optional)</Label>
            <Input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="What this variable is for" />
          </div>
          <div className="flex items-center gap-2">
            <Checkbox id="is_secret" checked={isSecret} onCheckedChange={(v) => setIsSecret(!!v)} />
            <Label htmlFor="is_secret" className="text-sm font-normal">Mark as secret (value will be masked in UI)</Label>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
          <Button onClick={handleSave} disabled={saving || !key.trim()}>
            {saving && <Loader2 className="h-4 w-4 animate-spin mr-1" />}
            {editingVar ? "Save" : "Create"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

export function GlobalVariablesCard() {
  const { variables, isLoading, fetchVariables, removeVariable } = useVariablesStore();
  const { toast } = useToast();
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingVar, setEditingVar] = useState<typeof variables[0] | null>(null);

  useEffect(() => {
    fetchVariables();
  }, [fetchVariables]);

  const handleDelete = async (id: number, key: string) => {
    try {
      await removeVariable(id);
      toast({ title: "Deleted", description: `Variable "${key}" removed` });
    } catch {
      toast({ title: "Error", description: "Failed to delete variable", variant: "destructive" });
    }
  };

  return (
    <>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <div>
            <CardTitle>Global Variables</CardTitle>
            <CardDescription>
              Key-value pairs reusable across all workflows via <code className="text-xs bg-muted px-1 rounded">{"{{vars.key}}"}</code>
            </CardDescription>
          </div>
          <Button size="sm" onClick={() => { setEditingVar(null); setDialogOpen(true); }}>
            <Plus className="h-4 w-4 mr-1" />
            Add Variable
          </Button>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : variables.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-8">
              No variables yet. Add one to get started.
            </p>
          ) : (
            <div className="border rounded-md">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left p-3 font-medium">Key</th>
                    <th className="text-left p-3 font-medium">Value</th>
                    <th className="text-left p-3 font-medium">Description</th>
                    <th className="w-20 p-3" />
                  </tr>
                </thead>
                <tbody>
                  {variables.map((v) => (
                    <tr key={v.id} className="border-b last:border-b-0">
                      <td className="p-3 font-mono text-xs">{v.key}</td>
                      <td className="p-3 font-mono text-xs">{v.is_secret ? "••••••••" : v.value}</td>
                      <td className="p-3 text-muted-foreground">{v.description || "—"}</td>
                      <td className="p-3">
                        <div className="flex gap-1">
                          <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => { setEditingVar(v); setDialogOpen(true); }}>
                            <Pencil className="h-3.5 w-3.5" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive" onClick={() => handleDelete(v.id, v.key)}>
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
      <AddEditDialog open={dialogOpen} onOpenChange={setDialogOpen} editingVar={editingVar} />
    </>
  );
}
