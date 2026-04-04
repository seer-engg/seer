import { useForm } from 'react-hook-form';
import { useEffect } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Form } from '@/components/ui/form';
import type { PostgresFormValues, DialogMode } from './hooks/usePostgresBinding';
import type { PostgresTestResponse, IntegrationResource } from '@/lib/api-client';
import { ConnectionStringInput } from './ConnectionStringInput';
import { IndividualFieldsInput } from './IndividualFieldsInput';
import { AccessModeField } from './AccessModeField';
import { TestResultDisplay } from './TestResultDisplay';
import { SSLModeField } from './SSLModeField';
import { NameAndMethodFields } from './NameAndMethodFields';
import { FormButtons } from './FormButtons';

const createPostgresFormSchema = (mode: DialogMode) => z.object({
  name: z.string().min(1, 'Name is required'),
  inputMode: z.enum(['connection_string', 'individual']),
  connectionString: z.string().optional(),
  host: z.string().optional(),
  port: z.coerce.number().min(1).max(65535).optional(),
  database: z.string().optional(),
  username: z.string().optional(),
  password: z.string().optional(),
  sslMode: z.enum(['disable', 'allow', 'prefer', 'require', 'verify-ca', 'verify-full']),
  accessMode: z.enum(['restricted', 'unrestricted']),
}).refine((data) => {
  // In edit mode, we don't require connection details since they're already saved
  if (mode === 'edit') {
    return true;
  }
  if (data.inputMode === 'connection_string') {
    return !!data.connectionString?.trim();
  }
  return !!data.host?.trim() && !!data.database?.trim();
}, {
  message: 'Please provide connection details',
  path: ['connectionString'],
});

const getFormDefaults = (
  isEditMode: boolean,
  resource?: IntegrationResource | null,
): PostgresFormValues => {
  if (isEditMode && resource) {
    const metadata = resource.metadata as {
      host?: string;
      port?: number;
      database?: string;
      ssl_mode?: string;
      access_mode?: string;
    } | undefined;

    return {
      name: resource.name || '',
      inputMode: 'individual',
      connectionString: '',
      host: metadata?.host || '',
      port: metadata?.port || 5432,
      database: metadata?.database || '',
      username: '',
      password: '',
      sslMode: (metadata?.ssl_mode as PostgresFormValues['sslMode']) || 'prefer',
      accessMode: (metadata?.access_mode as PostgresFormValues['accessMode']) || 'restricted',
    };
  }

  return {
    name: '',
    inputMode: 'connection_string',
    connectionString: '',
    host: '',
    port: 5432,
    database: '',
    username: '',
    password: '',
    sslMode: 'prefer',
    accessMode: 'restricted',
  };
};

interface PostgresCredentialsFormProps {
  onTest: (values: PostgresFormValues) => Promise<PostgresTestResponse>;
  onSave: (values: PostgresFormValues) => void;
  isTesting: boolean;
  isBinding: boolean;
  testResult: PostgresTestResponse | null;
  mode?: DialogMode;
  initialResource?: IntegrationResource | null;
}

export function PostgresCredentialsForm({
  onTest,
  onSave,
  isTesting,
  isBinding,
  testResult,
  mode = 'create',
  initialResource,
}: PostgresCredentialsFormProps) {
  const isEditMode = mode === 'edit';

  const form = useForm<PostgresFormValues>({
    resolver: zodResolver(createPostgresFormSchema(mode)),
    defaultValues: getFormDefaults(false, null),
  });

  useEffect(() => {
    form.reset(getFormDefaults(isEditMode, initialResource));
  }, [isEditMode, initialResource, form]);

  const inputMode = form.watch('inputMode');

  const handleTest = async () => {
    const isValid = await form.trigger();
    if (isValid) {
      await onTest(form.getValues());
    }
  };

  const handleSubmit = form.handleSubmit((values) => {
    onSave(values);
  });

  return (
    <Form {...form}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <NameAndMethodFields form={form} />

        {inputMode === 'connection_string' ? (
          <ConnectionStringInput form={form} isEditMode={isEditMode} />
        ) : (
          <IndividualFieldsInput form={form} isEditMode={isEditMode} />
        )}

        <SSLModeField form={form} />
        <AccessModeField form={form} />

        {testResult && <TestResultDisplay testResult={testResult} />}

        <FormButtons
          onTest={handleTest}
          isTesting={isTesting}
          isBinding={isBinding}
          isEditMode={isEditMode}
        />
      </form>
    </Form>
  );
}
