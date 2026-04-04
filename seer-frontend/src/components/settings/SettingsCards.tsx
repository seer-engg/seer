import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Shield,
  User as UserIcon,
  Server,
  LogOut,
} from "lucide-react";
import { ConnectionCard } from "./ConnectionCard";
import type { ConnectedAccount } from "@/lib/api-client";
import { useToast } from "@/hooks/utility/use-toast";
import { useBackendHealth } from "@/lib/backend-health";
import {
  getBackendBaseUrl,
  getDefaultBackendBaseUrl,
  getStoredBackendBaseUrl,
  setStoredBackendBaseUrl,
} from "@/lib/api-client";
import type { User } from "@clerk/clerk-react";
import { useSignOut } from "@/hooks/useAuthProvider";

type BackendSource = "query-param" | "custom" | "default";

const resolveActiveBackendSource = (): BackendSource => {
  if (typeof window !== "undefined") {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("backend")) {
      return "query-param";
    }
  }

  if (getStoredBackendBaseUrl()) {
    return "custom";
  }

  return "default";
};

const useBackendUrlSettings = () => {
  const { toast } = useToast();
  const defaultBackendUrl = getDefaultBackendBaseUrl();
  const storedBackendUrl = getStoredBackendBaseUrl();

  const [backendUrlMode, setBackendUrlMode] = useState<"default" | "custom">(
    storedBackendUrl ? "custom" : "default"
  );
  const [backendUrlInput, setBackendUrlInput] = useState<string>(storedBackendUrl ?? "");
  const [activeBackendUrl, setActiveBackendUrl] = useState<string>(getBackendBaseUrl());
  const [activeBackendSource, setActiveBackendSource] = useState<BackendSource>(resolveActiveBackendSource());

  useEffect(() => {
    setActiveBackendUrl(getBackendBaseUrl());
    setActiveBackendSource(resolveActiveBackendSource());
  }, []);

  const handleSaveBackendUrl = () => {
    if (!backendUrlInput.trim()) {
      toast({
        title: "Enter a valid URL",
        description: "Provide a backend API URL or use the cloud default.",
        variant: "destructive",
      });
      return;
    }

    const normalizedUrl = setStoredBackendBaseUrl(backendUrlInput);
    const resolvedUrl = normalizedUrl ?? getBackendBaseUrl();
    setBackendUrlMode("custom");
    setActiveBackendUrl(resolvedUrl);
    setActiveBackendSource(resolveActiveBackendSource());
    toast({
      title: "API URL updated",
      description: `Backend requests will use ${resolvedUrl}.`,
    });
  };

  const handleUseDefaultBackend = () => {
    setStoredBackendBaseUrl(null);
    const resolvedUrl = getBackendBaseUrl();
    setBackendUrlMode("default");
    setBackendUrlInput("");
    setActiveBackendUrl(resolvedUrl);
    setActiveBackendSource(resolveActiveBackendSource());
    toast({
      title: "Using cloud API",
      description: `Backend requests will use ${resolvedUrl}.`,
    });
  };

  return {
    backendUrlMode,
    backendUrlInput,
    defaultBackendUrl,
    activeBackendUrl,
    activeBackendSource,
    setBackendUrlInput,
    handleSaveBackendUrl,
    handleUseDefaultBackend,
  };
};

interface ProfileCardProps {
  user: User | null | undefined;
}

export function ProfileCard({ user }: ProfileCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <UserIcon className="h-5 w-5 text-seer" />
          </div>
          <div>
            <CardTitle className="text-base">Profile</CardTitle>
            <CardDescription>Your personal information</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label>Name</Label>
            <Input value={user?.fullName || ""} disabled className="bg-secondary/50" />
          </div>
          <div className="space-y-2">
            <Label>Email</Label>
            <Input
              value={user?.primaryEmailAddress?.emailAddress || ""}
              disabled
              className="bg-secondary/50"
            />
          </div>
        </div>
        <p className="text-xs text-muted-foreground">
          Profile information is managed through your connected account provider.
        </p>
      </CardContent>
    </Card>
  );
}

export function BackendUrlCard() {
  const {
    backendUrlMode,
    backendUrlInput,
    defaultBackendUrl,
    activeBackendUrl,
    activeBackendSource,
    setBackendUrlInput,
    handleSaveBackendUrl,
    handleUseDefaultBackend,
  } = useBackendUrlSettings();

  const { isHealthy } = useBackendHealth();
  const isSelfHostedBackend = activeBackendUrl !== defaultBackendUrl;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <Server className="h-5 w-5 text-seer" />
          </div>
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-base">Backend Location</CardTitle>
                {isSelfHostedBackend && isHealthy && (
                  <Badge
                    variant="secondary"
                    className="h-5 px-1.5 text-[10px] bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                  >
                    Self-Hosted
                  </Badge>
                )}
              </div>
              <CardDescription>Switch between Seer Cloud and your own backend</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Update the backend API base URL used by the app. Queries, tools, and workflow requests will use this URL.
          </p>

          <div className="space-y-2">
            <Label htmlFor="backend-url">Custom API URL</Label>
            <Input
              id="backend-url"
              value={backendUrlInput}
              placeholder="https://your-backend.example.com"
              onChange={(event) => setBackendUrlInput(event.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Default cloud URL: <span className="font-medium">{defaultBackendUrl}</span>
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <Button variant="outline" onClick={handleUseDefaultBackend} disabled={backendUrlMode === "default"}>
              Use cloud default
            </Button>
            <Button onClick={handleSaveBackendUrl} disabled={!backendUrlInput.trim()}>
              Save custom URL
            </Button>
          </div>

          <div className="text-xs text-muted-foreground">
            Active base URL: <span className="font-medium">{activeBackendUrl}</span>{" "}
            {activeBackendSource === "query-param"
              ? "(query parameter override)"
              : activeBackendSource === "custom"
                ? "(custom)"
                : "(cloud default)"}
          </div>
          {activeBackendSource === "query-param" && (
            <p className="text-xs text-muted-foreground">
              Remove the ?backend=... query parameter to switch away from this override.
            </p>
          )}
        </CardContent>
      </Card>
  );
}

interface ConnectionsCardProps {
  connections: ConnectedAccount[];
  connectionsLoading: boolean;
  deletingConnectionId: string | null;
  onDeleteClick: (conn: ConnectedAccount) => void;
}

export function ConnectionsCard({
  connections,
  connectionsLoading,
  deletingConnectionId,
  onDeleteClick,
}: ConnectionsCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
            <Shield className="h-5 w-5 text-seer" />
          </div>
          <div>
            <CardTitle className="text-base">Connected Accounts</CardTitle>
            <CardDescription>View your connected accounts and granted permissions</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {connectionsLoading ? (
          <div className="text-sm text-muted-foreground">Loading connections...</div>
        ) : connections.length === 0 ? (
          <div className="text-sm text-muted-foreground">
            No connected accounts found. Connect an account from a workflow to get started.
          </div>
        ) : (
          <div className="space-y-4">
            {connections.map((conn: ConnectedAccount, index: number) => (
              <ConnectionCard
                key={conn.id}
                connection={conn}
                isDeleting={deletingConnectionId === conn.id}
                isLastCard={index === connections.length - 1}
                onDelete={onDeleteClick}
              />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function LogoutCard() {
  const signOut = useSignOut();
  const { toast } = useToast();

  const handleSignOut = async () => {
    try {
      await signOut({ redirectUrl: "/sign-in" });
      toast({
        title: "Signed out successfully",
        description: "You have been logged out of your account.",
      });
    } catch (error) {
      toast({
        title: "Failed to sign out",
        description: "An error occurred while signing out.",
        variant: "destructive",
      });
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-destructive/10 flex items-center justify-center">
            <LogOut className="h-5 w-5 text-destructive" />
          </div>
          <div>
            <CardTitle className="text-base">Sign Out</CardTitle>
            <CardDescription>End your current session</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">
          You will be signed out and redirected to the sign-in page.
        </p>
        <Button variant="destructive" onClick={handleSignOut}>
          Sign Out
        </Button>
      </CardContent>
    </Card>
  );
}
