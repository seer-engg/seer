/**
 * SharedIntegrationsCard Component
 *
 * Card for managing OAuth connections shared with the team.
 * Shows both team-shared integrations and the user's personal connections
 * that can be shared.
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Separator } from '@/components/ui/separator';
import { Label } from '@/components/ui/label';
import { Share2, Unlink, Loader2, Users } from 'lucide-react';
import { useConnectionsQuery } from '@/hooks/useConnectionsQuery';
import { organizationApi } from '@/lib/organization-api';
import { useOrganizationStore } from '@/stores/organizationStore';
import { toast } from '@/components/ui/sonner';
import { DynamicIntegrationIcon } from '@/components/ui/DynamicIntegrationIcon';
import { useIntegrationMetadataStore } from '@/stores/integrationMetadataStore';
import type { SharedIntegration } from '@/types/organization';

/* eslint-disable max-lines-per-function */
export function SharedIntegrationsCard() {
  const [sharedIntegrations, setSharedIntegrations] = useState<SharedIntegration[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sharingId, setSharingId] = useState<string | null>(null);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const connectionsQuery = useConnectionsQuery();
  const connections = connectionsQuery.data ?? [];
  const connectionsLoaded = connectionsQuery.status === 'success';
  const getDisplayName = useIntegrationMetadataStore((s) => s.getDisplayName);

  const orgId = currentOrganization?.id;
  const isTeamOrg = currentOrganization?.type === 'team';

  // Fetch shared integrations
  useEffect(() => {
    if (!orgId || !isTeamOrg) return;

    const fetchSharedIntegrations = async () => {
      setIsLoading(true);
      try {
        const integrations = await organizationApi.getSharedIntegrations(orgId);
        setSharedIntegrations(integrations);
      } catch (error) {
        console.error('Failed to fetch shared integrations:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSharedIntegrations();
  }, [orgId, isTeamOrg]);

  // Don't show on personal workspace
  if (!currentOrganization || currentOrganization.type === 'personal') {
    return null;
  }

  // Get connections that are shared with this org
  const teamSharedConnections = sharedIntegrations.filter((i) => i.sharedWithOrg);

  // Get user's personal connections that can be shared
  const personalConnections = connections.filter(
    (c) => !sharedIntegrations.some((i) => i.id === c.id)
  );

  const handleShare = async (connectionId: string) => {
    if (!currentOrganization) return;

    setSharingId(connectionId);
    try {
      const sharedIntegration = await organizationApi.shareIntegration(
        currentOrganization.id,
        connectionId
      );
      setSharedIntegrations((prev) => [...prev, sharedIntegration]);
      toast.success('Integration shared with team');
    } catch (error) {
      console.error('Failed to share integration:', error);
      toast.error('Failed to share integration');
    } finally {
      setSharingId(null);
    }
  };

  const handleUnshare = async (connectionId: string) => {
    if (!currentOrganization) return;

    setSharingId(connectionId);
    try {
      await organizationApi.unshareIntegration(currentOrganization.id, connectionId);
      setSharedIntegrations((prev) => prev.filter((i) => i.id !== connectionId));
      toast.success('Integration unshared');
    } catch (error) {
      console.error('Failed to unshare integration:', error);
      toast.error('Failed to unshare integration');
    } finally {
      setSharingId(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Share2 className="h-5 w-5" />
          Shared Integrations
        </CardTitle>
        <CardDescription>
          OAuth connections shared with all team members
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Team Shared Integrations */}
        <div>
          <Label className="mb-3 block text-sm font-medium">Team Integrations</Label>
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-full" />
            </div>
          ) : teamSharedConnections.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center border rounded-lg border-dashed">
              No integrations shared with this team yet
            </p>
          ) : (
            <div className="space-y-2">
              {teamSharedConnections.map((integration) => (
                <div
                  key={integration.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <DynamicIntegrationIcon
                      integrationType={integration.provider}
                      width={32}
                      height={32}
                    />
                    <div>
                      <p className="font-medium">
                        {getDisplayName(integration.provider)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {integration.accountEmail || integration.accountName}
                        {integration.sharedByName && (
                          <> &middot; Shared by {integration.sharedByName}</>
                        )}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleUnshare(integration.id)}
                    disabled={sharingId === integration.id}
                  >
                    {sharingId === integration.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Unlink className="h-4 w-4" />
                        Unshare
                      </>
                    )}
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>

        <Separator />

        {/* Personal Connections to Share */}
        <div>
          <Label className="mb-2 block text-sm font-medium">Your Connections</Label>
          <p className="text-sm text-muted-foreground mb-3">
            Share your connections to let team members use them
          </p>

          {!connectionsLoaded ? (
            <div className="space-y-2">
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-full" />
            </div>
          ) : personalConnections.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4 text-center border rounded-lg border-dashed">
              No personal connections available to share
            </p>
          ) : (
            <div className="space-y-2">
              {personalConnections.map((connection) => (
                <div
                  key={connection.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <DynamicIntegrationIcon
                      integrationType={connection.provider}
                      width={32}
                      height={32}
                    />
                    <div>
                      <p className="font-medium">
                        {getDisplayName(connection.provider)}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {connection.toolkit?.slug || 'Connected'}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleShare(connection.id)}
                    disabled={sharingId === connection.id}
                  >
                    {sharingId === connection.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Users className="h-4 w-4" />
                        Share with Team
                      </>
                    )}
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
