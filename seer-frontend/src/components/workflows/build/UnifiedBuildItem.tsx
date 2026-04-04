import { useState, forwardRef } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { Wrench } from 'lucide-react';
import { cn } from '@/lib/utils';
import { getToolLogoUrl, getIntegrationLogoUrl } from '@/lib/logo-utils';
import {
  GoogleDriveIcon,
  GoogleSheetsIcon,
  GoogleDocsIcon,
  GmailIcon,
  GoogleCalendarIcon,
  GoogleSlidesIcon,
  LinkedInIcon,
  GitHubIcon,
  DiscordIcon,
  SlackIcon,
} from '@/components/icons/google-products';

import type { UnifiedItem } from '../buildtypes';

interface UnifiedBuildItemProps {
  item: UnifiedItem;
  onItemClick?: (item: UnifiedItem) => void;
}

function ActionIcon({ toolName }: { toolName: string }) {
  const [imgError, setImgError] = useState(false);
  const logoUrl = getToolLogoUrl(toolName, 32);

  if (!logoUrl || imgError) {
    return <Wrench className="w-4 h-4 text-muted-foreground" />;
  }

  return (
    <img
      src={logoUrl}
      alt={`${toolName} logo`}
      className="w-4 h-4 object-contain rounded-sm"
      onError={() => setImgError(true)}
    />
  );
}

/**
 * Custom SVG icons for integrations where Logo.dev doesn't work well:
 * - Google products: Logo.dev can't distinguish sub-products (all return "G" logo)
 * - Other common integrations: Ensures consistent, reliable icons
 */
const CUSTOM_INTEGRATION_ICONS: Record<string, React.ComponentType<React.SVGProps<SVGSVGElement>>> = {
  // Google products
  gmail: GmailIcon,
  google_drive: GoogleDriveIcon,
  google_sheets: GoogleSheetsIcon,
  google_docs: GoogleDocsIcon,
  google_calendar: GoogleCalendarIcon,
  google_slides: GoogleSlidesIcon,
  // Other common integrations
  linkedin: LinkedInIcon,
  github: GitHubIcon,
  discord: DiscordIcon,
  slack: SlackIcon,
};

export function IntegrationIcon({ integrationType }: { integrationType: string }) {
  const [imgError, setImgError] = useState(false);
  const normalizedType = integrationType.toLowerCase().trim();

  // Use custom SVG for known integrations
  const CustomIcon = CUSTOM_INTEGRATION_ICONS[normalizedType];
  if (CustomIcon) {
    return <CustomIcon className="w-4 h-4" />;
  }

  // Use Logo.dev for all other integrations
  const logoUrl = getIntegrationLogoUrl(integrationType);

  if (!logoUrl || imgError) {
    return <Wrench className="w-4 h-4 text-muted-foreground" />;
  }

  return (
    <img
      src={logoUrl}
      alt={`${integrationType} logo`}
      className="w-4 h-4 object-contain rounded-sm"
      onError={() => setImgError(true)}
    />
  );
}

function getDragData(item: UnifiedItem) {
  if (item.type === 'block') {
    return { type: 'block', blockType: item.blockType, label: item.label };
  }
  if (item.type === 'trigger') {
    return { type: 'trigger', triggerKey: item.triggerKey, title: item.label };
  }
  if (item.type === 'action' && item.integrationGroup) {
    // New: drag integration group
    const defaultTool = item.integrationGroup.tools[0];
    return {
      type: 'tool-group',
      integrationGroup: {
        integration_type: item.integrationGroup.integration_type,
        display_name: item.integrationGroup.display_name,
        tools: item.integrationGroup.tools,
        defaultTool: {
          name: defaultTool.name,
          provider: defaultTool.provider,
          output_schema: defaultTool.output_schema,
        },
      },
    };
  }
  if (item.type === 'action' && item.tool) {
    // Legacy: drag individual tool
    return {
      type: 'tool',
      tool: {
        name: item.tool.name,
        slug: item.tool.slug,
        provider: item.tool.provider,
        integration_type: item.tool.integration_type,
        output_schema: item.tool.output_schema,
      },
    };
  }
  return null;
}

const SimpleCard = forwardRef<HTMLDivElement, {
  item: UnifiedItem;
  onDragStart: (e: React.DragEvent) => void;
  onClick: () => void;
}>(({ item, onDragStart, onClick }, ref) => {
  // Generate data-testid based on item type
  const getTestId = () => {
    if (item.type === 'block') {
      return `block-${item.blockType}`;
    }
    if (item.type === 'action' && item.tool) {
      return `tool-${item.tool.name}`;
    }
    if (item.type === 'action' && item.integrationGroup) {
      return `tool-group-${item.integrationGroup.integration_type}`;
    }
    return undefined;
  };

  return (
    <Card
      ref={ref}
      data-testid={getTestId()}
      draggable
      onDragStart={onDragStart}
      className="relative cursor-grab active:cursor-grabbing hover:bg-accent transition-colors"
      onClick={onClick}
    >
      <CardContent className="p-1.5">
        <div className="flex items-center justify-between gap-1.5">
          <div className="flex items-center gap-1.5 min-w-0 flex-1">
            <div className="w-4 h-4 flex-shrink-0">
              {item.type === 'action' && item.tool ? (
                <ActionIcon toolName={item.tool.name} />
              ) : item.type === 'action' && item.integrationGroup ? (
                <IntegrationIcon integrationType={item.integrationGroup.integration_type} />
              ) : (
                item.icon
              )}
            </div>
            <p className="text-sm font-medium truncate">{item.label}</p>
          </div>
          {item.type === 'action' && item.integrationGroup && (
            <div className="flex items-center gap-1.5 flex-shrink-0">
              <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">
                {item.toolCount} {item.toolCount === 1 ? 'tool' : 'tools'}
              </Badge>
              {item.integrationGroup.isConnected && (
                <Badge variant="default" className="h-5 px-1.5 text-[10px] bg-green-500/10 text-green-700 dark:text-green-400 border-green-500/20">
                  Connected
                </Badge>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
});
SimpleCard.displayName = 'SimpleCard';

function StatusBadge({ status, badge }: { status?: string; badge: string }) {
  const badgeClass = status === 'ready'
    ? 'border-emerald-500/30 text-emerald-600'
    : status === 'action-required'
      ? 'border-amber-500/40 text-amber-600'
      : '';

  return (
    <Badge variant="outline" className={cn('text-[10px] flex-shrink-0', badgeClass)}>
      {badge}
    </Badge>
  );
}

function TriggerActions({ item }: { item: UnifiedItem }) {
  if (!item.secondaryActionLabel && !item.disabledReason) {
    return null;
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {item.secondaryActionLabel && item.onSecondaryAction && (
        <Button
          size="sm"
          variant="outline"
          disabled={item.isSecondaryActionLoading}
          onClick={(event) => {
            event.stopPropagation();
            item.onSecondaryAction?.();
          }}
        >
          {item.secondaryActionLabel}
        </Button>
      )}
      {item.disabledReason && (
        <span className="text-xs text-muted-foreground">{item.disabledReason}</span>
      )}
    </div>
  );
}

const TriggerCard = forwardRef<HTMLDivElement, {
  item: UnifiedItem;
  onDragStart: (e: React.DragEvent) => void;
  onClick: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}>(({ item, onDragStart, onClick, onKeyDown }, ref) => {
  // Generate data-testid for trigger
  const testId = item.type === 'trigger' && item.triggerKey ? `trigger-${item.triggerKey}` : undefined;

  return (
    <Card
      ref={ref}
      data-testid={testId}
      draggable={!item.disabled}
      onDragStart={onDragStart}
      role="button"
      tabIndex={item.disabled ? -1 : 0}
      aria-disabled={item.disabled}
      onClick={onClick}
      onKeyDown={onKeyDown}
      className={cn(
        'relative cursor-grab active:cursor-grabbing hover:bg-accent transition-colors',
        item.disabled && 'opacity-80 grayscale-[20%] cursor-not-allowed'
      )}
    >
      <CardContent className="flex flex-col gap-1.5 p-1.5">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5">
            <div className="w-4 h-4 flex-shrink-0">{item.icon}</div>
            <p className="text-sm font-medium truncate">{item.label}</p>
          </div>
          {item.badge && <StatusBadge status={item.status} badge={item.badge} />}
        </div>
        <TriggerActions item={item} />
      </CardContent>
    </Card>
  );
});
TriggerCard.displayName = 'TriggerCard';

export function UnifiedBuildItem({ item, onItemClick }: UnifiedBuildItemProps) {
  const handleDragStart = (e: React.DragEvent) => {
    if (item.type === 'trigger' && item.disabled) {
      e.preventDefault();
      return;
    }
    e.dataTransfer.effectAllowed = 'move';
    const data = getDragData(item);
    if (data) {
      e.dataTransfer.setData('application/reactflow', JSON.stringify(data));
    }
  };

  const handleClick = () => {
    if (item.type === 'trigger' && (item.disabled || item.isPrimaryActionLoading)) {
      return;
    }
    onItemClick?.(item);
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleClick();
    }
  };

  const cardContent = item.type === 'block' || item.type === 'action'
    ? <SimpleCard item={item} onDragStart={handleDragStart} onClick={handleClick} />
    : <TriggerCard item={item} onDragStart={handleDragStart} onClick={handleClick} onKeyDown={handleKeyDown} />;

  if (item.description || (item.type === 'action' && item.integrationGroup)) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{cardContent}</TooltipTrigger>
        <TooltipContent className={item.type === 'trigger' ? 'max-w-xs text-sm' : ''}>
          {item.type === 'action' && item.integrationGroup ? (
            <div className="space-y-1">
              <p className="font-medium">{item.description}</p>
              <p className="text-xs text-muted-foreground">
                Tools: {item.integrationGroup.tools.map((t) => t.name).join(', ')}
              </p>
            </div>
          ) : (
            <p>{item.description}</p>
          )}
        </TooltipContent>
      </Tooltip>
    );
  }

  return cardContent;
}
