import {
  FormField,
  FormItem,
  FormLabel,
  FormControl,
  FormMessage,
} from '@/components/ui/form';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { UseFormReturn } from 'react-hook-form';
import type { PostgresFormValues } from './hooks/usePostgresBinding';

interface SSLModeFieldProps {
  form: UseFormReturn<PostgresFormValues>;
}

export function SSLModeField({ form }: SSLModeFieldProps) {
  return (
    <FormField
      control={form.control}
      name="sslMode"
      render={({ field }) => (
        <FormItem>
          <FormLabel>SSL Mode</FormLabel>
          <Select onValueChange={field.onChange} defaultValue={field.value}>
            <FormControl>
              <SelectTrigger>
                <SelectValue placeholder="Select SSL mode" />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              <SelectItem value="disable">Disable</SelectItem>
              <SelectItem value="allow">Allow</SelectItem>
              <SelectItem value="prefer">Prefer (default)</SelectItem>
              <SelectItem value="require">Require</SelectItem>
              <SelectItem value="verify-ca">Verify CA</SelectItem>
              <SelectItem value="verify-full">Verify Full</SelectItem>
            </SelectContent>
          </Select>
          <FormMessage />
        </FormItem>
      )}
    />
  );
}
