import { CSSProperties, memo } from 'react';
import { GitBranch } from 'lucide-react';
import { Handle, NodeProps, Position, type Node } from '@xyflow/react';
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

export const IfElseBlockNode = memo(function IfElseBlockNode(
  props: NodeProps<WorkflowNode>
) {
  const orientation = useHandleOrientation();
  const isVertical = orientation === 'vertical';
  const trueHandle = getHandleStyles(isVertical, 35);
  const falseHandle = getHandleStyles(isVertical, 65);
  const labelContainerStyle = getLabelContainerStyle(isVertical);

  return (
    <BaseBlockNode
      {...props}
      icon={<GitBranch className="w-4 h-4 text-orange-500" />}
      color="orange"
      handles={{
        inputs: ['input'],
        outputs: [],
      }}
      outputButtons={
        <>
          <OutputEdgeButton
            nodeId={props.id}
            nodeType="if_else"
            branch="true"
            position={trueHandle.buttonStyle}
          />
          <OutputEdgeButton
            nodeId={props.id}
            nodeType="if_else"
            branch="false"
            position={falseHandle.buttonStyle}
          />
        </>
      }
    >
      <div className="h-4">
        <Handle
          id="true"
          type="source"
          position={trueHandle.position}
          style={{
            position: 'absolute',
            ...trueHandle.handleStyle,
          }}
          className="!w-3 !h-3 !bg-orange-500 !border-2 !border-background"
        />
        <Handle
          id="false"
          type="source"
          position={falseHandle.position}
          style={{
            position: 'absolute',
            ...falseHandle.handleStyle,
          }}
          className="!w-3 !h-3 !bg-muted-foreground !border-2 !border-background"
        />
        <div
          className="pointer-events-none absolute inset-y-0 text-[11px] text-muted-foreground"
          style={labelContainerStyle}
        >
          <span
            className={cn('absolute rounded bg-muted px-2 py-0.5 text-xs')}
            style={trueHandle.labelStyle}
          >
            True
          </span>
          <span
            className={cn('absolute rounded bg-muted px-2 py-0.5 text-xs')}
            style={falseHandle.labelStyle}
          >
            False
          </span>
        </div>
      </div>
    </BaseBlockNode>
  );
});
