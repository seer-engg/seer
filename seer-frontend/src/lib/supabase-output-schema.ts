/**
 * Generates the standard Supabase event payload schema.
 * These are the fields available from Supabase database events.
 */
export interface SupabaseOutputField {
  name: string;
  description: string;
}

export function generateSupabaseOutputSchema(): SupabaseOutputField[] {
  return [
    {
      name: 'id',
      description: 'Unique event identifier',
    },
    {
      name: 'old_record',
      description: 'Previous record state (UPDATE/DELETE only)',
    },
    {
      name: 'new_record',
      description: 'New or updated record (INSERT/UPDATE only)',
    },
    {
      name: 'changes',
      description: 'Array of changed column names',
    },
    {
      name: 'commit_timestamp',
      description: 'Event timestamp (ISO 8601 format)',
    },
    {
      name: 'type',
      description: 'Event type: INSERT, UPDATE, or DELETE',
    },
    {
      name: 'schema',
      description: 'Database schema name',
    },
    {
      name: 'table',
      description: 'Table name',
    },
  ];
}
