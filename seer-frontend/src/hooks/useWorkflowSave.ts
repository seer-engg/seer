import { useState, useCallback } from 'react';
import { useWorkflowActions } from './useWorkflowActions';
import { toast } from '@/components/ui/sonner';
import { useActiveWorkflowId } from './useActiveWorkflowId';

export function useWorkflowSave() {
  const [isSaving, setIsSaving] = useState(false);
  const activeWorkflowId = useActiveWorkflowId();
  const { handleSave } = useWorkflowActions();

  const saveWorkflow = useCallback(async () => {
    if (!activeWorkflowId) {
      return;
    }

    setIsSaving(true);
    try {
      await handleSave();
    } catch (error) {
      console.error('Failed to save workflow before OAuth:', error);
      toast.error('Failed to save workflow', {
        description: 'Unable to save workflow before connecting. Please try again.',
      });
      throw error;
    } finally {
      setIsSaving(false);
    }
  }, [activeWorkflowId, handleSave]);

  return {
    saveWorkflow,
    isSaving,
    hasWorkflow: Boolean(activeWorkflowId),
  };
}
