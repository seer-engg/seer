import { Database, Plus, Trash2, Server, Pencil } from 'lucide-react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { listPostgresBindings, type IntegrationResource } from '@/lib/api-client';
import { postgresKeys } from '@/lib/query-keys';
import { PostgresBindingDialog } from '@/components/workflows/dialogs/ResourcePicker/PostgresBindingDialog';
import { usePostgresBinding } from '@/components/workflows/dialogs/ResourcePicker/hooks/usePostgresBinding';
import { DeleteDatabaseDialog } from './DeleteDatabaseDialog';
import { EmptyDatabasesState } from './EmptyDatabasesState';
import { useDatabaseDeletion } from './useDatabaseDeletion';

function DatabaseRow({
  resource,
  onEdit,
  onDelete,
}: {
  resource: IntegrationResource;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const metadata = resource.metadata as { host?: string; database?: string } | undefined;
  const hostInfo = metadata?.host
    ? `${metadata.host}${metadata.database ? `/${metadata.database}` : ''}`
    : null;

  return (
    <div className="flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-seer/10 flex items-center justify-center">
          <Server className="h-4 w-4 text-seer" />
        </div>
        <div>
          <p className="text-sm font-medium">{resource.name || 'Unnamed Database'}</p>
          {hostInfo && (
            <p className="text-xs text-muted-foreground line-clamp-1">{hostInfo}</p>
          )}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
            resource.status === 'active'
              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-muted text-muted-foreground'
          }`}
        >
          {resource.status}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-foreground"
          onClick={onEdit}
        >
          <Pencil className="h-4 w-4" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted-foreground hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

function CardHeaderContent({ isLoading }: { isLoading?: boolean }) {
  return (
    <CardHeader>
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-seer/10 flex items-center justify-center">
          <Database className="h-5 w-5 text-seer" />
        </div>
        <div className="flex-1">
          <CardTitle className="text-base">Databases</CardTitle>
          <CardDescription>
            {isLoading ? 'Loading...' : 'Manage your connected PostgreSQL databases'}
          </CardDescription>
        </div>
      </div>
    </CardHeader>
  );
}

export function DatabasesTab() {
  const queryClient = useQueryClient();
  const bindingState = usePostgresBinding();
  const { bindModalOpen, setBindModalOpen, openCreateDialog, openEditDialog } = bindingState;
  const deletion = useDatabaseDeletion();

  const { data: bindings = [], isLoading } = useQuery({
    queryKey: postgresKeys.bindings(),
    queryFn: listPostgresBindings,
  });

  const handleBindSuccess = () => setBindModalOpen(false);
  const refetchResources = () => queryClient.invalidateQueries({ queryKey: postgresKeys.bindings() });

  if (isLoading) {
    return (
      <Card>
        <CardHeaderContent isLoading />
      </Card>
    );
  }

  return (
    <>
      <Card>
        <CardHeaderContent />
        <CardContent className="space-y-4">
          {bindings.length === 0 ? (
            <EmptyDatabasesState />
          ) : (
            <div className="space-y-1">
              {bindings.map((resource) => (
                <DatabaseRow
                  key={resource.id}
                  resource={resource}
                  onEdit={() => openEditDialog(resource)}
                  onDelete={() => deletion.handleDeleteClick(resource)}
                />
              ))}
            </div>
          )}
          <Button onClick={openCreateDialog} className="w-full" variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Add Database
          </Button>
        </CardContent>
      </Card>

      <PostgresBindingDialog
        open={bindModalOpen}
        onOpenChange={setBindModalOpen}
        bindingState={bindingState}
        onBindSuccess={handleBindSuccess}
        refetchResources={refetchResources}
      />

      <DeleteDatabaseDialog
        open={deletion.deleteDialogOpen}
        onOpenChange={deletion.setDeleteDialogOpen}
        resource={deletion.resourceToDelete}
        onConfirm={deletion.handleConfirmDelete}
        isPending={deletion.isPending}
      />
    </>
  );
}
