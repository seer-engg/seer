import { useMemo, useCallback } from 'react';
import type { InputDef } from '@/types/workflow-spec';

/**
  * Workflow inputs have been removed from the persisted spec.
  * This hook now returns empty defaults and no-op handlers to keep
  * existing components stable while the UI migrates away from inputs.
  */
export function useWorkflowInputs() {
  const workflowInputsDef = useMemo<Record<string, InputDef>>(() => ({}), []);

  const handleWorkflowInputsChange = useCallback(async () => {
    // Inputs are deprecated; intentionally left as a no-op.
  }, []);

  const inputFields = useMemo(() => [], []);

  return {
    workflowInputsDef,
    handleWorkflowInputsChange,
    inputFields,
  };
}
