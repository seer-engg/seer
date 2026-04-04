/**
 * TeamSettingsCard Component
 *
 * Card displaying team information and settings.
 * Only shown when the user is in a team organization context.
 */

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Building2, Loader2, Pencil, Check, X } from 'lucide-react';
import { RoleBadge } from './RoleBadge';
import { useOrganizationStore } from '@/stores/organizationStore';
import { toast } from '@/components/ui/sonner';

/* eslint-disable max-lines-per-function */
export function TeamSettingsCard() {
  const [isEditing, setIsEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const currentOrganization = useOrganizationStore((s) => s.currentOrganization);
  const updateOrganization = useOrganizationStore((s) => s.updateOrganization);
  const canManageBilling = useOrganizationStore((s) => s.canManageBilling);

  if (!currentOrganization || currentOrganization.type === 'personal') {
    return null;
  }

  const handleStartEdit = () => {
    setEditName(currentOrganization.name);
    setIsEditing(true);
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditName('');
  };

  const handleSaveEdit = async () => {
    if (!editName.trim() || editName.trim() === currentOrganization.name) {
      handleCancelEdit();
      return;
    }

    setIsSaving(true);
    try {
      await updateOrganization(currentOrganization.id, { name: editName.trim() });
      toast.success('Team name updated');
      setIsEditing(false);
    } catch {
      toast.error('Failed to update team name');
    } finally {
      setIsSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      handleCancelEdit();
    }
  };

  return (
    <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Building2 className="h-5 w-5" />
            Team Information
          </CardTitle>
          <CardDescription>
            Manage your team settings
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <Label className="text-xs text-muted-foreground">Team Name</Label>
              {isEditing ? (
                <div className="flex items-center gap-2 mt-1">
                  <Input
                    value={editName}
                    onChange={(e) => setEditName(e.target.value)}
                    onKeyDown={handleKeyDown}
                    className="h-8 text-sm"
                    autoFocus
                    disabled={isSaving}
                  />
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8"
                    onClick={handleSaveEdit}
                    disabled={isSaving}
                  >
                    {isSaving ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Check className="h-4 w-4" />
                    )}
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="h-8 w-8"
                    onClick={handleCancelEdit}
                    disabled={isSaving}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ) : (
                <div className="flex items-center gap-2 mt-1">
                  <p className="text-lg font-medium">{currentOrganization.name}</p>
                  {canManageBilling() && (
                    <Button
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7"
                      onClick={handleStartEdit}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>

          <div>
            <Label className="text-xs text-muted-foreground">Your Role</Label>
            <div className="mt-1">
              <RoleBadge role={currentOrganization.role} />
            </div>
          </div>

          {currentOrganization.memberCount !== undefined && (
            <div>
              <Label className="text-xs text-muted-foreground">Members</Label>
              <p className="text-sm mt-1">
                {currentOrganization.memberCount} member{currentOrganization.memberCount !== 1 && 's'}
              </p>
            </div>
          )}

        </CardContent>
      </Card>
    );
}
