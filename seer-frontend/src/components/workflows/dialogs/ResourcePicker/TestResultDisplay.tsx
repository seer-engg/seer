import { CheckCircle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { PostgresTestResponse } from '@/lib/api-client';

interface TestResultDisplayProps {
  testResult: PostgresTestResponse;
}

export function TestResultDisplay({ testResult }: TestResultDisplayProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-md border p-3 text-sm',
        testResult.connected
          ? 'border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950 dark:text-green-200'
          : 'border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950 dark:text-red-200',
      )}
    >
      {testResult.connected ? (
        <>
          <CheckCircle className="h-4 w-4 flex-shrink-0" />
          <span>
            Connected successfully
            {testResult.server_version && ` (PostgreSQL ${testResult.server_version})`}
          </span>
        </>
      ) : (
        <>
          <XCircle className="h-4 w-4 flex-shrink-0" />
          <span>{testResult.error || 'Connection failed'}</span>
        </>
      )}
    </div>
  );
}
