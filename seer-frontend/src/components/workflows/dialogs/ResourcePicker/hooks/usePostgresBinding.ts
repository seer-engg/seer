import { useState, useCallback } from 'react';
import { useToast } from '@/hooks/utility/use-toast';
import {
  testPostgresConnection,
  bindPostgresDatabase,
  updatePostgresDatabase,
  type PostgresTestRequest,
  type PostgresBindRequest,
  type PostgresTestResponse,
  type IntegrationResource,
} from '@/lib/api-client';

export interface PostgresFormValues {
  name: string;
  inputMode: 'connection_string' | 'individual';
  connectionString?: string;
  host?: string;
  port?: number;
  database?: string;
  username?: string;
  password?: string;
  sslMode: 'disable' | 'allow' | 'prefer' | 'require' | 'verify-ca' | 'verify-full';
  accessMode: 'restricted' | 'unrestricted';
}

export type DialogMode = 'create' | 'edit';

const buildConnectionParams = (values: PostgresFormValues): PostgresTestRequest => {
  if (values.inputMode === 'connection_string' && values.connectionString) {
    return {
      connection_string: values.connectionString,
      ssl_mode: values.sslMode,
    };
  }
  return {
    host: values.host,
    port: values.port || 5432,
    database: values.database,
    username: values.username,
    password: values.password,
    ssl_mode: values.sslMode,
  };
};

const buildBindParams = (values: PostgresFormValues): PostgresBindRequest => ({
  ...buildConnectionParams(values),
  name: values.name,
  access_mode: values.accessMode,
});

const resetDialogState = (
  setBindModalOpen: (open: boolean) => void,
  setTestResult: (result: PostgresTestResponse | null) => void,
  setResourceToEdit: (resource: IntegrationResource | null) => void,
  setDialogMode: (mode: DialogMode) => void,
  isUpdate: boolean,
) => {
  setBindModalOpen(false);
  setTestResult(null);
  if (isUpdate) {
    setResourceToEdit(null);
    setDialogMode('create');
  }
};

const testConnection = async (
  values: PostgresFormValues,
  setTestResult: (result: PostgresTestResponse | null) => void,
  toast: ReturnType<typeof useToast>['toast'],
): Promise<PostgresTestResponse> => {
  try {
    const result = await testPostgresConnection(buildConnectionParams(values));
    setTestResult(result);
    toast({
      title: result.connected ? 'Connection successful' : 'Connection failed',
      description: result.connected
        ? result.server_version
          ? `Connected to PostgreSQL ${result.server_version}`
          : 'Database connection verified'
        : result.error || 'Unable to connect to database',
      variant: result.connected ? 'default' : 'destructive',
    });
    return result;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Connection test failed';
    const failedResult = { success: false, error: errorMessage };
    setTestResult(failedResult);
    toast({
      title: 'Connection failed',
      description: errorMessage,
      variant: 'destructive',
    });
    return failedResult;
  }
};

export function usePostgresBinding() {
  const [bindModalOpen, setBindModalOpen] = useState(false);
  const [isBinding, setIsBinding] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<PostgresTestResponse | null>(null);
  const [dialogMode, setDialogMode] = useState<DialogMode>('create');
  const [resourceToEdit, setResourceToEdit] = useState<IntegrationResource | null>(null);

  const { toast } = useToast();

  const handleTestConnection = useCallback(async (values: PostgresFormValues) => {
    setIsTesting(true);
    setTestResult(null);
    try {
      return await testConnection(values, setTestResult, toast);
    } finally {
      setIsTesting(false);
    }
  }, [toast]);

  const saveDatabase = useCallback(async (
    values: PostgresFormValues,
    onSuccess: (resource: IntegrationResource) => void,
    refetch: () => void,
    isUpdate: boolean,
  ) => {
    setIsBinding(true);
    try {
      const params = buildBindParams(values);
      const response = isUpdate && resourceToEdit
        ? await updatePostgresDatabase(resourceToEdit.id, params)
        : await bindPostgresDatabase(params);

      toast({
        title: isUpdate ? 'Database updated' : 'Database connected',
        description: `${values.name} has been ${isUpdate ? 'updated' : 'added'} successfully`,
      });
      onSuccess(response.resource);
      refetch();
      resetDialogState(setBindModalOpen, setTestResult, setResourceToEdit, setDialogMode, isUpdate);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : `Failed to ${isUpdate ? 'update' : 'bind'} database`;
      toast({
        title: `Failed to ${isUpdate ? 'update' : 'connect'} database`,
        description: errorMessage,
        variant: 'destructive',
      });
    } finally {
      setIsBinding(false);
    }
  }, [toast, resourceToEdit]);

  const handleBindDatabase = useCallback(
    (values: PostgresFormValues, onSuccess: (resource: IntegrationResource) => void, refetch: () => void) =>
      saveDatabase(values, onSuccess, refetch, false),
    [saveDatabase],
  );

  const handleUpdateDatabase = useCallback(
    (values: PostgresFormValues, onSuccess: (resource: IntegrationResource) => void, refetch: () => void) =>
      saveDatabase(values, onSuccess, refetch, true),
    [saveDatabase],
  );

  const openDialog = useCallback((mode: DialogMode, resource: IntegrationResource | null = null) => {
    setResourceToEdit(resource);
    setDialogMode(mode);
    setTestResult(null);
    setBindModalOpen(true);
  }, []);

  const openEditDialog = useCallback(
    (resource: IntegrationResource) => openDialog('edit', resource),
    [openDialog],
  );

  const openCreateDialog = useCallback(() => openDialog('create'), [openDialog]);

  return {
    bindModalOpen,
    setBindModalOpen,
    isBinding,
    isTesting,
    testResult,
    dialogMode,
    resourceToEdit,
    handleTestConnection,
    handleBindDatabase,
    handleUpdateDatabase,
    openEditDialog,
    openCreateDialog,
  };
}
