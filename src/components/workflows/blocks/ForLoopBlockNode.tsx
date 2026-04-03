import { CSSProperties, memo } from 'react';
import { Repeat } from 'lucide-react';
import { Handle, Position, NodeProps, type Node } from '@xyflow/react';
import { cn } from '@/lib/utils';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';
import { OutputEdgeButton } from '../canvas/OutputEdgeButton';
import { useHandleOrientation } from '../canvas/hooks/useHandleOrientation';

type WorkflowNode = Node<WorkflowNodeData>;
type HandleStyles = {
  position: Position;
  handleStyle: CSSProperties;
  labelStyle: CSSProperties;
  buttonStyle: CSSProperties;
};

const getHandleStyles = (isVertical: boolean, offsetPercent: number): HandleStyles => ({
  position: isVertical ? Position.Bottom : Position.Right,
  handleStyle: isVertical
    ? { bottom: -8, left: `${offsetPercent}%`, transform: 'translateX(-50%)' }
    : { top: `${offsetPercent}%`, right: -8 },
  labelStyle: isVertical
    ? { bottom: '-28px', left: `${offsetPercent}%`, transform: 'translateX(-50%)' }
    : { top: `${offsetPercent}%`, right: 'calc(-8px - 0.25rem)', transform: 'translateY(-50%)' },
  buttonStyle: isVertical
    ? { bottom: '-42px', left: `${offsetPercent}%`, transform: 'translateX(-50%)' }
    : { top: `${offsetPercent}%`, right: '-48px', transform: 'translateY(-50%)' },
});

const getLabelContainerStyle = (isVertical: boolean): CSSProperties =>
  isVertical ? { bottom: '-8px', left: 0, right: 0 } : { right: 'calc(-8px - 0.25rem)' };

export const ForLoopBlockNode = memo(function ForLoopBlockNode(
  props: NodeProps<WorkflowNode>
) {
  const orientation = useHandleOrientation();
  const isVertical = orientation === 'vertical';
  const loopHandle = getHandleStyles(isVertical, 35);
  const exitHandle = getHandleStyles(isVertical, 65);
  const config = props.data.config || {};
  const literalItems = Array.isArray(config.array_literal) ? config.array_literal : [];
  const literalPreview = literalItems.slice(0, 3).join(', ');
  const hasLegacyLiteral = literalItems.length > 0;
  const labelContainerStyle = getLabelContainerStyle(isVertical);

  return (
    <BaseBlockNode
      {...props}
      icon={<Repeat className="w-4 h-4 text-green-500" />}
      color="green"
      handles={{
        inputs: ['input'],
        outputs: [],
      }}
      outputButtons={
        <>
          <OutputEdgeButton
            nodeId={props.id}
            nodeType="for_loop"
            branch="loop"
            position={loopHandle.buttonStyle}
          />
          <OutputEdgeButton
            nodeId={props.id}
            nodeType="for_loop"
            branch="exit"
            position={exitHandle.buttonStyle}
          />
        </>
      }
    >
      {hasLegacyLiteral && (
        <p className="mt-2 text-[11px] text-muted-foreground">
          Legacy list: {literalPreview}
          {literalItems.length > 3 ? '…' : ''}
        </p>
      )}

      <div className="mt-2 h-4">
        <Handle
          id="loop"
          type="source"
          position={loopHandle.position}
          style={{
            position: 'absolute',
            ...loopHandle.handleStyle,
          }}
          className="!w-3 !h-3 !bg-green-500 !border-2 !border-background"
        />
        <Handle
          id="exit"
          type="source"
          position={exitHandle.position}
          style={{
            position: 'absolute',
            ...exitHandle.handleStyle,
          }}
          className="!w-3 !h-3 !bg-muted-foreground !border-2 !border-background"
        />
        <div
          className="pointer-events-none absolute inset-y-0 text-[11px] text-muted-foreground"
          style={labelContainerStyle}
        >
          <span
            className={cn('absolute rounded bg-muted px-2 py-0.5 text-xs')}
            style={loopHandle.labelStyle}
          >
            Loop
          </span>
          <span
            className={cn('absolute rounded bg-muted px-2 py-0.5 text-xs')}
            style={exitHandle.labelStyle}
          >
            Exit
          </span>
        </div>
      </div>

    </BaseBlockNode>
  );
});
