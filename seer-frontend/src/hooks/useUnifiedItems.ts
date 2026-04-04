import { useMemo, createElement } from 'react';
import type { UnifiedItem, BuiltInBlock, Tool } from '@/components/workflows/buildtypes';
import type { TriggerListOption } from '@/components/workflows/build/TriggerSection';
import { TRIGGER_ICON_BY_KEY, TRIGGER_COLOR_BY_KEY } from '@/components/workflows/triggers/constants';
import { IntegrationIcon } from '@/components/workflows/build/UnifiedBuildItem';
import { useToolingData } from './useToolingData';

function getTriggerIntegrationType(triggerKey: string): string | null {
  const parts = triggerKey.split('.');
  if (parts.length >= 3) return parts[1];
  return null;
}

/**
 * Transform all data sources (blocks, triggers, tools) into unified items
 */
export function useUnifiedItems(
  blocks: BuiltInBlock[],
  triggerOptions: TriggerListOption[],
  tools: Tool[] // kept for compatibility but not used
): UnifiedItem[] {
  const { integrationGroups } = useToolingData();
  return useMemo<UnifiedItem[]>(() => {
    const items: UnifiedItem[] = [];

    // Add blocks
    blocks.forEach((block) => {
      items.push({
        id: `block-${block.type}`,
        type: 'block',
        label: block.label,
        description: block.description,
        icon: block.icon,
        blockType: block.type,
        builtInBlock: block,
        searchTerms: `${block.label} ${block.description}`.toLowerCase(),
      });
    });

    // Add triggers
    triggerOptions.forEach((trigger) => {
      const Icon = TRIGGER_ICON_BY_KEY[trigger.key];
      const colors = TRIGGER_COLOR_BY_KEY[trigger.key];
      const iconColorClass = colors?.text ?? 'text-primary';
      const integType = getTriggerIntegrationType(trigger.key);
      const icon = integType
        ? createElement(IntegrationIcon, { integrationType: integType })
        : Icon
          ? createElement(Icon, { className: `w-4 h-4 ${iconColorClass}` })
          : createElement('span', { className: 'text-xs font-semibold' }, 'T');
      items.push({
        id: `trigger-${trigger.key}`,
        type: 'trigger',
        label: trigger.title,
        description: trigger.description || undefined,
        icon,
        triggerKey: trigger.key,
        status: trigger.status,
        badge: trigger.badge,
        actionLabel: trigger.actionLabel,
        secondaryActionLabel: trigger.secondaryActionLabel,
        disabled: trigger.disabled,
        disabledReason: trigger.disabledReason,
        onPrimaryAction: trigger.onPrimaryAction,
        onSecondaryAction: trigger.onSecondaryAction,
        isPrimaryActionLoading: trigger.isPrimaryActionLoading,
        isSecondaryActionLoading: trigger.isSecondaryActionLoading,
        searchTerms: `${trigger.title} ${trigger.description || ''}`.toLowerCase(),
      });
    });

    // Add actions (integration groups)
    // Note: icons are rendered dynamically via IntegrationIcon component
    integrationGroups.forEach((group) => {
      const toolNames = group.tools.map((t) => t.name).join(' ');
      const toolDescriptions = group.tools.map((t) => t.description || '').join(' ');
      items.push({
        id: `action-${group.integration_type}`,
        type: 'action',
        label: group.display_name,
        description: group.description,
        integrationType: group.integration_type,
        integrationGroup: {
          integration_type: group.integration_type,
          display_name: group.display_name,
          tools: group.tools,
          isConnected: group.isConnected,
        },
        toolCount: group.tools.length,
        searchTerms: `${group.display_name} ${group.description} ${toolNames} ${toolDescriptions}`.toLowerCase(),
      });
    });

    return items;
  }, [blocks, triggerOptions, integrationGroups]);
}
