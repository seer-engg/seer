import { Shield, ShieldAlert } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormDescription,
} from '@/components/ui/form';
import type { UseFormReturn } from 'react-hook-form';
import type { PostgresFormValues } from './hooks/usePostgresBinding';

interface AccessModeFieldProps {
  form: UseFormReturn<PostgresFormValues>;
}

export function AccessModeField({ form }: AccessModeFieldProps) {
  const accessMode = form.watch('accessMode');

  return (
    <FormField
      control={form.control}
      name="accessMode"
      render={({ field }) => (
        <FormItem>
          <FormLabel>Access Mode</FormLabel>
          <FormControl>
            <Tabs value={field.value} onValueChange={field.onChange} className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="restricted" className="gap-2">
                  <Shield className="h-4 w-4" />
                  Restricted
                </TabsTrigger>
                <TabsTrigger value="unrestricted" className="gap-2">
                  <ShieldAlert className="h-4 w-4" />
                  Unrestricted
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </FormControl>
          <FormDescription>
            {accessMode === 'restricted' ? (
              <span className="text-muted-foreground">
                Read-only access: SELECT, EXPLAIN, and SHOW queries only
              </span>
            ) : (
              <span className="text-amber-600 dark:text-amber-500">
                Full SQL access including INSERT, UPDATE, DELETE, and DDL statements
              </span>
            )}
          </FormDescription>
        </FormItem>
      )}
    />
  );
}
