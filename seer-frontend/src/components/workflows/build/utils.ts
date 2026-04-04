import type { ItemType } from '../buildtypes';

export function getLayoutClass(type: ItemType): string {
  if (type === 'block') {
    return 'grid grid-cols-2 gap-1.5';
  }
  return 'space-y-1.5';
}

export function getTypeLabel(type: ItemType): string {
  switch (type) {
    case 'block':
      return 'Blocks';
    case 'trigger':
      return 'Triggers';
    case 'action':
      return 'Actions';
    default:
      return '';
  }
}
