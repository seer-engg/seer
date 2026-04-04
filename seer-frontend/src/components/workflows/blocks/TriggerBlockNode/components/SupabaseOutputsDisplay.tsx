import { generateSupabaseOutputSchema } from '@/lib/supabase-output-schema';
import { Label } from '@/components/ui/label';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';

export interface SupabaseOutputsDisplayProps {
  isConfigured: boolean;
}

export function SupabaseOutputsDisplay({
  isConfigured,
}: SupabaseOutputsDisplayProps) {
  if (!isConfigured) {
    return (
      <p className="text-xs text-muted-foreground">
        Configure a project, schema, and table to see available outputs.
      </p>
    );
  }

  const outputs = generateSupabaseOutputSchema();

  return (
    <div className="space-y-2">
      <Label className="text-xs uppercase text-muted-foreground">
        Available Outputs
      </Label>
      <div className="space-y-1.5 rounded-md border border-dashed bg-muted/40 p-3">
        <TooltipProvider>
          {outputs.map(({ name, description }) => (
            <div
              key={name}
              className="flex items-center gap-2"
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-xs font-medium text-foreground cursor-help hover:text-seer">
                    {name}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="right" className="text-xs max-w-xs">
                  {description}
                </TooltipContent>
              </Tooltip>
            </div>
          ))}
        </TooltipProvider>
      </div>
    </div>
  );
}
