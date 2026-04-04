import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { DollarSign, Save, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/utility/use-toast";
import { getUserSettings, updateUserSettings } from "@/lib/api-client";
import { userKeys } from "@/lib/query-keys";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

function CostCapCardHeader() {
  return (
    <div className="flex items-center gap-3">
      <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
        <DollarSign className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
      </div>
      <div>
        <CardTitle className="text-base">Per-Run Cost Cap</CardTitle>
        <CardDescription>Maximum LLM cost per execution</CardDescription>
      </div>
    </div>
  );
}

interface CostCapInputProps {
  currentCap: number;
  costCapInput: string;
  setCostCapInput: (value: string) => void;
  handleSave: () => void;
  isPending: boolean;
}

function CostCapInput({ currentCap, costCapInput, setCostCapInput, handleSave, isPending }: CostCapInputProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="cost-cap" className="text-sm font-medium">
        Maximum cost per execution (USD)
      </Label>
      <div className="flex gap-2">
        <div className="relative flex-1">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm">
            $
          </span>
          <Input
            id="cost-cap"
            type="number"
            min="0.10"
            max="1000"
            step="0.50"
            placeholder={currentCap.toFixed(2)}
            value={costCapInput}
            onChange={(e) => setCostCapInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                handleSave();
              }
            }}
            className="pl-6"
            disabled={isPending}
          />
        </div>
        <Button
          onClick={handleSave}
          disabled={!costCapInput || isPending}
          size="default"
        >
          <Save className="h-4 w-4 mr-2" />
          {isPending ? "Saving..." : "Save"}
        </Button>
      </div>
    </div>
  );
}

function CostCapInfoBox({ currentCap }: { currentCap: number }) {
  return (
    <div className="rounded-lg bg-muted/50 p-3 space-y-2">
      <div className="flex items-start gap-2">
        <AlertCircle className="h-4 w-4 text-muted-foreground mt-0.5 flex-shrink-0" />
        <div className="text-xs text-muted-foreground space-y-1">
          <p className="font-medium text-foreground">
            Current cap: <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
              ${currentCap.toFixed(2)}
            </span>
          </p>
          <p>Execution stops immediately when this cap is reached</p>
          <p>Applies separately to each chat thread and workflow run</p>
          <p>Does not affect your monthly subscription credits</p>
        </div>
      </div>
    </div>
  );
}

export function CostCapCard() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: settings, isLoading } = useQuery({
    queryKey: userKeys.settings(),
    queryFn: getUserSettings,
  });

  const [costCapInput, setCostCapInput] = useState<string>("");

  const updateMutation = useMutation({
    mutationFn: (newCap: number) =>
      updateUserSettings({ per_run_cost_cap_usd: newCap }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      toast({
        title: "Cost cap updated successfully",
        description: `New cap: $${parseFloat(costCapInput).toFixed(2)}`,
      });
      setCostCapInput("");
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to update cost cap",
        description: error.message,
        variant: "destructive"
      });
    },
  });

  const currentCap = (settings?.preferences?.per_run_cost_cap_usd as number | undefined) ?? 5.0;

  const handleSave = () => {
    const parsed = parseFloat(costCapInput);

    if (isNaN(parsed) || parsed < 0.1 || parsed > 1000) {
      toast({
        title: "Invalid value",
        description: "Cost cap must be between $0.10 and $1000.00",
        variant: "destructive"
      });
      return;
    }

    updateMutation.mutate(parsed);
  };

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-emerald-500/10 flex items-center justify-center">
              <DollarSign className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
            </div>
            <div>
              <CardTitle className="text-base">Per-Run Cost Cap</CardTitle>
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
        <CostCapCardHeader />
      </CardHeader>

      <CardContent className="space-y-4">
        <CostCapInput
          currentCap={currentCap}
          costCapInput={costCapInput}
          setCostCapInput={setCostCapInput}
          handleSave={handleSave}
          isPending={updateMutation.isPending}
        />
        <CostCapInfoBox currentCap={currentCap} />
      </CardContent>
    </Card>
  );
}
