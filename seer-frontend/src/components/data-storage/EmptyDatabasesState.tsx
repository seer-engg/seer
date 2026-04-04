import { Database } from 'lucide-react';

export function EmptyDatabasesState() {
  return (
    <div className="py-6 text-center">
      <Database className="h-10 w-10 text-muted-foreground/40 mx-auto mb-3" />
      <p className="text-sm text-muted-foreground">No databases connected</p>
      <p className="text-xs text-muted-foreground mt-1">
        Add a PostgreSQL database to use with your workflows
      </p>
    </div>
  );
}
