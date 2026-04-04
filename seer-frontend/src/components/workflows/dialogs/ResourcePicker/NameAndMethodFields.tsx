import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
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

interface NameAndMethodFieldsProps {
  form: UseFormReturn<PostgresFormValues>;
}

export function NameAndMethodFields({ form }: NameAndMethodFieldsProps) {
  return (
    <>
      <FormField
        control={form.control}
        name="name"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Name</FormLabel>
            <FormControl>
              <Input placeholder="My Production Database" {...field} />
            </FormControl>
            <FormDescription>A friendly name to identify this database</FormDescription>
            <FormMessage />
          </FormItem>
        )}
      />

      <FormField
        control={form.control}
        name="inputMode"
        render={({ field }) => (
          <FormItem>
            <FormLabel>Connection Method</FormLabel>
            <FormControl>
              <Tabs value={field.value} onValueChange={field.onChange} className="w-full">
                <TabsList className="grid w-full grid-cols-2">
                  <TabsTrigger value="connection_string">Connection String</TabsTrigger>
                  <TabsTrigger value="individual">Individual Fields</TabsTrigger>
                </TabsList>
              </Tabs>
            </FormControl>
          </FormItem>
        )}
      />
    </>
  );
}
