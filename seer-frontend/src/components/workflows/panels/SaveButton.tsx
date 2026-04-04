import { Node } from '@xyflow/react';
import { Button } from '@/components/ui/button';
import { Save } from 'lucide-react';
import { toast } from '@/components/ui/sonner';
import { WorkflowNodeData, WorkflowNodeUpdateOptions } from '../types';
import { ToolBlockConfig } from '@/components/workflows/block-config/types';
import { ToolMetadata } from '../block-config';
import { buildPersistedNodeUpdate } from './hooks/persistedNodeConfig';

// Helper function to check if Supabase binding is required
function validateSupabaseBinding(
  toolSchema: ToolMetadata | undefined,
  config: ToolBlockConfig,
): { isValid: boolean; error?: string } {
  const requiresSupabaseBinding = toolSchema?.integration_type === 'supabase';
  if (requiresSupabaseBinding && !config?.params?.integration_resource_id) {
    return {
      isValid: false,
      error: 'Supabase project required',
    };
  }
  return { isValid: true };
}

// Helper function to handle save with validation
async function handleConfigSave(params: {
  node: Node<WorkflowNodeData>;
  toolSchema: ToolMetadata | undefined;
  config: ToolBlockConfig;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
}): Promise<void> {
  const { node, toolSchema, config, onUpdate } = params;

  const validation = validateSupabaseBinding(toolSchema, config);
  if (!validation.isValid) {
    toast.error(validation.error!, {
      description: 'Select and bind a Supabase project before saving this tool.',
    });
    return;
  }

  const updatePayload = buildPersistedNodeUpdate({
    originalConfig: (node.data.config || {}) as ToolBlockConfig,
    currentConfig: config,
  });

  if (!updatePayload) {
    return;
  }

  try {
    await onUpdate(
      node.id,
      updatePayload,
      { persist: true },
    );
    toast.success('Configuration saved', {
      description: 'Block configuration has been saved successfully',
      duration: 2000,
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Unable to save block configuration';
    toast.error('Failed to save configuration', {
      description: message,
    });
  }
}

export interface SaveButtonProps {
  visible: boolean;
  node: Node<WorkflowNodeData>;
  toolSchema?: ToolMetadata;
  config: ToolBlockConfig;
  onUpdate: (
    nodeId: string,
    updates: Partial<WorkflowNodeData>,
    options?: WorkflowNodeUpdateOptions,
  ) => Promise<void> | void;
}

export function SaveButton({
  visible,
  node,
  toolSchema,
  config,
  onUpdate,
}: SaveButtonProps): JSX.Element | null {
  if (!visible) return null;

  return (
    <Button
      onClick={() =>
        void handleConfigSave({
          node,
          toolSchema,
          config,
          onUpdate,
        })
      }
      variant="brand"
      className="w-full"
      size="sm"
    >
      <Save className="w-4 h-4 mr-2" />
      Save Configuration
    </Button>
  );
}
