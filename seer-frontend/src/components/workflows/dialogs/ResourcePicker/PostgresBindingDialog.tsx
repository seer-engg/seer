import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import type { IntegrationResource } from '@/lib/api-client';
import type { usePostgresBinding } from './hooks/usePostgresBinding';
import { PostgresCredentialsForm } from './PostgresCredentialsForm';

export interface PostgresBindingDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bindingState: ReturnType<typeof usePostgresBinding>;
  onBindSuccess: (resource: IntegrationResource) => void;
  refetchResources: () => void;
}

export function PostgresBindingDialog({
  open,
  onOpenChange,
  bindingState,
  onBindSuccess,
  refetchResources,
}: PostgresBindingDialogProps) {
  const {
    isBinding,
    isTesting,
    testResult,
    dialogMode,
    resourceToEdit,
    handleTestConnection,
    handleBindDatabase,
    handleUpdateDatabase,
  } = bindingState;

  const isEditMode = dialogMode === 'edit';
  const title = isEditMode ? 'Edit Postgres Database' : 'Add Postgres Database';
  const description = isEditMode
    ? 'Update your database settings. Leave password blank to keep current credentials.'
    : 'Enter your database credentials to connect. Your credentials are encrypted and stored securely.';

  const handleSave = isEditMode
    ? (values: Parameters<typeof handleBindDatabase>[0]) =>
        handleUpdateDatabase(values, onBindSuccess, refetchResources)
    : (values: Parameters<typeof handleBindDatabase>[0]) =>
        handleBindDatabase(values, onBindSuccess, refetchResources);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>

        <PostgresCredentialsForm
          onTest={handleTestConnection}
          onSave={handleSave}
          isTesting={isTesting}
          isBinding={isBinding}
          testResult={testResult}
          mode={dialogMode}
          initialResource={resourceToEdit}
        />
      </DialogContent>
    </Dialog>
  );
}
