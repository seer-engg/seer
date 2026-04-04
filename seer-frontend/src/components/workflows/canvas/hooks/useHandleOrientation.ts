import type { CSSProperties } from 'react';
import { Position } from '@xyflow/react';
import { useUIStore } from '@/stores/uiStore';

export type HandleOrientation = 'vertical' | 'horizontal';

export function useHandleOrientation(): HandleOrientation {
  return useUIStore((state) => state.handleOrientation);
}

export function getHandlePositionProps(
  orientation: HandleOrientation,
  kind: 'input' | 'output',
): { position: Position; style: CSSProperties } {
  const isVertical = orientation === 'vertical';

  if (kind === 'input') {
    return {
      position: isVertical ? Position.Top : Position.Left,
      style: isVertical
        ? { top: -8, left: '50%', transform: 'translateX(-50%)' }
        : { left: -8, top: '50%', transform: 'translateY(-50%)' },
    };
  }

  return {
    position: isVertical ? Position.Bottom : Position.Right,
    style: isVertical
      ? { bottom: -8, left: '50%', transform: 'translateX(-50%)' }
      : { right: -8, top: '50%', transform: 'translateY(-50%)' },
  };
}

export function getDefaultOutputButtonStyle(orientation: HandleOrientation): CSSProperties {
  return orientation === 'vertical'
    ? { bottom: '-48px', left: '50%', transform: 'translateX(-50%)' }
    : { top: '50%', right: '-48px', transform: 'translateY(-50%)' };
}
