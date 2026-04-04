import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Globe, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/utility/use-toast";
import { getUserSettings, updateUserSettings } from "@/lib/api-client";
import { userKeys } from "@/lib/query-keys";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { TIMEZONE_OPTIONS } from "@/components/workflows/triggers/constants";

export function TimezoneCard() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: settings, isLoading } = useQuery({
    queryKey: userKeys.settings(),
    queryFn: getUserSettings,
  });

  const browserTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  const [selectedTimezone, setSelectedTimezone] = useState<string>("");

  useEffect(() => {
    if (settings) {
      setSelectedTimezone(settings.timezone ?? browserTimezone ?? "America/Chicago");
    }
  }, [settings, browserTimezone]);

  const updateMutation = useMutation({
    mutationFn: (timezone: string) => updateUserSettings({ timezone }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.settings() });
      toast({
        title: "Timezone updated",
        description: `Timezone set to ${selectedTimezone}`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to update timezone",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleSave = () => {
    updateMutation.mutate(selectedTimezone);
  };

  const isDirty = settings && selectedTimezone !== (settings.timezone ?? browserTimezone ?? "America/Chicago");

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
              <Globe className="h-5 w-5 text-seer" />
            </div>
            <div>
              <CardTitle className="text-base">Timezone</CardTitle>
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
            <Globe className="h-5 w-5 text-seer" />
          </div>
          <div>
            <CardTitle className="text-base">Timezone</CardTitle>
            <CardDescription>Used for scheduling and display</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="timezone-select">Timezone</Label>
          <div className="flex gap-2">
            <select
              id="timezone-select"
              value={selectedTimezone}
              onChange={(e) => setSelectedTimezone(e.target.value)}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              {TIMEZONE_OPTIONS.map((tz) => (
                <option key={tz} value={tz}>
                  {tz}
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
          {!settings?.timezone && (
            <p className="text-xs text-muted-foreground">
              Auto-detected from your browser: {browserTimezone}
            </p>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
