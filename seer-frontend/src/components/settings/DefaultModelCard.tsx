import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Bot, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/utility/use-toast";
import { getUserSettings, updateUserSettings } from "@/lib/api-client";
import { userKeys, modelKeys } from "@/lib/query-keys";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { ModelInfo } from "@/components/workflows/buildtypes";
import { backendApiClient } from "@/lib/api-client";

export function DefaultModelCard() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: settings, isLoading: isLoadingSettings } = useQuery({
    queryKey: userKeys.settings(),
    queryFn: getUserSettings,
  });

  const { data: models = [], isLoading: isLoadingModels } = useQuery<ModelInfo[]>({
    queryKey: modelKeys.available(),
    queryFn: () => backendApiClient.request<ModelInfo[]>('/api/models', { method: 'GET' }),
  });

  const [selectedModel, setSelectedModel] = useState<string>("");

  useEffect(() => {
    if (settings) {
      const saved = settings.preferences?.default_model as string | undefined;
      setSelectedModel(saved || "qwen/qwen3-235b-a22b-2507");
    }
  }, [settings]);

  const updateMutation = useMutation({
    mutationFn: (modelId: string) =>
      updateUserSettings({ preferences: { default_model: modelId } }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      toast({ title: "Default model updated", description: `Now using ${selectedModel}` });
    },
    onError: (error: Error) => {
      toast({ title: "Failed to update default model", description: error.message, variant: "destructive" });
    },
  });

  const handleSave = () => {
    updateMutation.mutate(selectedModel);
  };

  const savedModel = (settings?.preferences?.default_model as string) || "qwen/qwen3-235b-a22b-2507";
  const isDirty = selectedModel !== savedModel;
  const isLoading = isLoadingSettings || isLoadingModels;

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
              <Bot className="h-5 w-5 text-seer" />
            </div>
            <div>
              <CardTitle className="text-base">Default Model</CardTitle>
              <CardDescription>Loading...</CardDescription>
            </div>
          </div>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <Bot className="h-5 w-5 text-seer" />
          </div>
          <div>
            <CardTitle className="text-base">Default Model</CardTitle>
            <CardDescription>Used for new workflow nodes and chat</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="default-model-select">Model</Label>
          <div className="flex gap-2">
            <select
              id="default-model-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
            <Button
              onClick={handleSave}
              disabled={!isDirty || updateMutation.isPending}
              size="default"
            >
              <Save className="h-4 w-4 mr-2" />
              {updateMutation.isPending ? "Saving..." : "Save"}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
