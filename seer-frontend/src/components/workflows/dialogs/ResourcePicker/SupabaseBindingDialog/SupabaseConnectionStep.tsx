import { Button } from '@/components/ui/button';

export interface SupabaseConnectionStepProps {
  onConnect: () => void;
}

export function SupabaseConnectionStep({
  onConnect,
}: SupabaseConnectionStepProps) {
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Connect your Supabase management account to browse projects.
      </p>
      <Button onClick={onConnect}>Connect Supabase</Button>
    </div>
  );
}
