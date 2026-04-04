import { memo } from 'react';
import { ImageIcon } from 'lucide-react';
import { NodeProps } from '@xyflow/react';
import { WorkflowNodeData } from '../types';
import { BaseBlockNode } from './BaseBlockNode';

export const ImageGenBlockNode = memo(function ImageGenBlockNode(
  props: NodeProps<WorkflowNodeData>
) {
  return (
    <BaseBlockNode
      {...props}
      icon={<ImageIcon className="w-4 h-4 text-pink-500" />}
      color="pink"
      consumesCredits
      handles={{
        inputs: ['input'],
        outputs: ['output'],
      }}
    />
  );
});
