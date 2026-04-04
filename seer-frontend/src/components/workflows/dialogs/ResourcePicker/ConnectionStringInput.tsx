import { Textarea } from '@/components/ui/textarea';
import {
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
  FormDescription,
} from '@/components/ui/form';
import type { UseFormReturn } from 'react-hook-form';
import type { PostgresFormValues } from './hooks/usePostgresBinding';

interface ConnectionStringInputProps {
  form: UseFormReturn<PostgresFormValues>;
  isEditMode: boolean;
}

export function ConnectionStringInput({ form, isEditMode }: ConnectionStringInputProps) {
  return (
    <FormField
      control={form.control}
      name="connectionString"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Connection String</FormLabel>
          <FormControl>
            <Textarea
              placeholder={
                isEditMode
                  ? 'Enter new connection string to update credentials'
                  : 'postgresql://user:password@host:5432/database'
              }
              className="font-mono text-sm"
              rows={2}
              {...field}
            />
          </FormControl>
          <FormDescription>
            {isEditMode
              ? 'Leave blank to keep existing credentials, or enter a new connection string'
              : 'Standard PostgreSQL connection string format'}
          </FormDescription>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
