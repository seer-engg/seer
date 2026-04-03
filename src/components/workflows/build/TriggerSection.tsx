import type { ReactNode } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

import { TRIGGER_ICON_BY_KEY, TRIGGER_COLOR_BY_KEY } from '@/components/workflows/triggers/constants';

export interface TriggerListOption {
  key: string;
  title: string;
  description?: string | null;
  disabled?: boolean;
  disabledReason?: string;
  badge?: string;
  status?: 'ready' | 'action-required';
  actionLabel?: string;
  secondaryActionLabel?: string;
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
  isPrimaryActionLoading?: boolean;
  isSecondaryActionLoading?: boolean;
}

interface TriggerSectionProps {
  options: TriggerListOption[];
  isLoading?: boolean;
  emptyState?: ReactNode;
  infoMessage?: string;
}

const TriggerHeader = () => <h3 className="text-sm font-medium text-left mb-2">Triggers</h3>;

const LoadingState = () => (
  <div>
    <div className="mb-2 flex items-center justify-between">
      <h3 className="text-sm font-medium text-left">Triggers</h3>
      <span className="text-xs text-muted-foreground">Loading…</span>
    </div>
    <Skeleton className="h-24 w-full rounded-lg" />
  </div>
);

const EmptyState = ({ state }: { state?: ReactNode }) => (
  <div>
    <TriggerHeader />
    <div className="rounded-md border border-dashed p-4 text-sm text-muted-foreground">
      {state || 'No triggers available for this workspace yet.'}
    </div>
  </div>
);

export function TriggerSection({
  options,
  isLoading = false,
  emptyState,
  infoMessage,
}: TriggerSectionProps) {
  if (isLoading) return <LoadingState />;
  if (!options.length) return <EmptyState state={emptyState} />;

  const TriggerCard = ({ option }: { option: TriggerListOption }) => {
    const Icon = TRIGGER_ICON_BY_KEY[option.key];
    const colors = TRIGGER_COLOR_BY_KEY[option.key] ?? { bg: 'bg-primary/10', text: 'text-primary' };
    const disabled = option.disabled ?? Boolean(option.disabledReason);

    const handleDragStart = (e: React.DragEvent) => {
      if (disabled) e.preventDefault();
      else {
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('application/reactflow', JSON.stringify({
          type: 'trigger', triggerKey: option.key, title: option.title,
        }));
      }
    };

    const handleClick = () => {
      if (!disabled && !option.isPrimaryActionLoading) option.onPrimaryAction?.();
    };

    const cardEl = (
      <Card
        key={option.key}
        draggable={!disabled}
        onDragStart={handleDragStart}
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-disabled={disabled}
        onClick={handleClick}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled && !option.isPrimaryActionLoading) {
            e.preventDefault();
            option.onPrimaryAction?.();
          }
        }}
        className={cn(
          'border border-border/70 bg-background transition-colors',
          disabled ? 'opacity-80 grayscale-[20%] cursor-not-allowed' : 'cursor-grab active:cursor-grabbing hover:border-primary/50',
        )}
      >
        <CardContent className="flex flex-col gap-3 p-1">
          <div className="flex items-start justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className={cn('rounded-md p-2', colors.bg, colors.text)}>
                {Icon ? <Icon className="h-4 w-4" /> : <span className="text-xs font-semibold">T</span>}
              </div>
              <p className="text-sm font-medium">{option.title}</p>
            </div>
            {option.badge && (
              <Badge variant="outline" className={cn('text-[10px]',
                option.status === 'ready' ? 'border-emerald-500/30 text-emerald-600' :
                option.status === 'action-required' ? 'border-amber-500/40 text-amber-600' : '',
              )}>
                {option.badge}
              </Badge>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {option.secondaryActionLabel && option.onSecondaryAction && (
              <Button
                size="sm"
                variant="outline"
                disabled={option.isSecondaryActionLoading}
                onClick={(e) => { e.stopPropagation(); option.onSecondaryAction?.(); }}
              >
                {option.secondaryActionLabel}
              </Button>
            )}
            {option.disabledReason && <span className="text-xs text-muted-foreground">{option.disabledReason}</span>}
          </div>
        </CardContent>
      </Card>
    );

    return option.description ? (
      <Tooltip key={option.key}>
        <TooltipTrigger asChild>{cardEl}</TooltipTrigger>
        <TooltipContent className="max-w-xs text-sm">{option.description}</TooltipContent>
      </Tooltip>
    ) : (
      cardEl
    );
  };

  return (
    <div>
      <h3 className="text-sm font-medium text-left mb-2">Triggers</h3>
      {infoMessage && (
        <div className="mb-3 rounded-md border border-dashed px-3 py-2 text-xs text-muted-foreground bg-muted/40">
          {infoMessage}
        </div>
      )}
      <div className="space-y-2">
        {options.map((option) => <TriggerCard key={option.key} option={option} />)}
      </div>
    </div>
  );
}

