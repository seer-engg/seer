import { Video, CheckCircle2 } from 'lucide-react';

export function MeetingsEADetailsSection() {
  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Video className="h-4 w-4" />
        meetingsEA Bot Events
      </div>

      <p className="text-xs text-muted-foreground">
        Triggers when a meetingsEA bot changes state (joined, recording, ended) or when a transcript is updated.
      </p>

      <div className="flex items-center gap-1.5">
        <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
        <span className="text-xs text-green-600 font-medium">Active</span>
      </div>

      <div className="pt-2 border-t border-dashed border-border/60">
        <p className="text-xs text-muted-foreground">
          This trigger fires automatically when bots created by the <code className="text-xs font-mono bg-muted px-1 rounded">meetings_ea_create_bot</code> tool change state. No additional setup required.
        </p>
      </div>
    </div>
  );
}
