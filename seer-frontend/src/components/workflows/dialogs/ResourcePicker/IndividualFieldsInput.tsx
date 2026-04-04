import { Input } from '@/components/ui/input';
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

interface IndividualFieldsInputProps {
  form: UseFormReturn<PostgresFormValues>;
  isEditMode: boolean;
}

export function IndividualFieldsInput({ form, isEditMode }: IndividualFieldsInputProps) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-3 gap-3">
        <FormField
          control={form.control}
          name="host"
          render={({ field }) => (
            <FormItem className="col-span-2">
              <FormLabel>Host</FormLabel>
              <FormControl>
                <Input placeholder="localhost" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="port"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Port</FormLabel>
              <FormControl>
                <Input type="number" placeholder="5432" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
      </div>
      <FormField
        control={form.control}
        name="database"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Database</FormLabel>
            <FormControl>
              <Input placeholder="postgres" {...field} />
            </FormControl>
            <FormMessage />
          </FormItem>
        )}
      />
      <div className="grid grid-cols-2 gap-3">
        <FormField
          control={form.control}
          name="username"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Username</FormLabel>
              <FormControl>
                <Input placeholder="postgres" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="password"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Password</FormLabel>
              <FormControl>
                <Input
                  type="password"
                  placeholder={isEditMode ? 'Leave blank to keep current' : '********'}
                  {...field}
                />
              </FormControl>
              {isEditMode && (
                <FormDescription>
                  Leave blank to keep existing password
                </FormDescription>
              )}
              <FormMessage />
            </FormItem>
          )}
        />
      </div>
    </div>
  );
}
