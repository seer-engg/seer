import type { UnifiedItem, ItemType } from '@/components/workflows/buildtypes';

export interface ItemGroup {
  type: ItemType;
  items: UnifiedItem[];
}

/**
 * Group consecutive items by type for layout
 */
export function groupConsecutiveItemsByType(
  items: UnifiedItem[]
): ItemGroup[] {
  const groups: ItemGroup[] = [];
  let currentGroup: ItemGroup | null = null;

  items.forEach((item) => {
    if (!currentGroup || currentGroup.type !== item.type) {
      currentGroup = { type: item.type, items: [item] };
      groups.push(currentGroup);
    } else {
      currentGroup.items.push(item);
    }
  });

  return groups;
}
