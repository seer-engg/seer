import { NodeViewWrapper, type NodeViewProps } from '@tiptap/react';

export function TemplateVariableView({ node }: NodeViewProps) {
  return (
    <NodeViewWrapper
      as="span"
      className="inline rounded bg-primary/10 text-primary px-1 py-0.5 font-mono text-[0.85em] whitespace-nowrap"
    >
      {`\${${node.attrs.path}}`}
    </NodeViewWrapper>
  );
}
