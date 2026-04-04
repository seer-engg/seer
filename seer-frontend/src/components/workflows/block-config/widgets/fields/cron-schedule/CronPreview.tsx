import { describeCronExpression } from './utils';

interface CronPreviewProps {
  expression: string;
}

export function CronPreview({ expression }: CronPreviewProps) {
  return (
    <div className="pt-2 border-t border-dashed">
      <p className="text-xs text-muted-foreground">
        <span className="font-medium">Schedule:</span> {describeCronExpression(expression)}
      </p>
      <p className="text-xs font-mono text-muted-foreground/70 mt-0.5">
        {expression}
      </p>
    </div>
  );
}
