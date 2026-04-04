import { useCallback, useEffect, useMemo, useState } from 'react';
import type { WorkflowNodeData } from '../../../types';
import {
  BindingState,
  buildBindingsPayload,
  buildDefaultBindingState,
  deriveBindingStateFromSubscription,
} from '../../../triggers/utils';
import {
  copyToClipboard,
  deleteTrigger,
  quickInsertBinding,
  toggleTrigger,
  updateBindingMode,
  updateBindingValue,
} from './handlers';

// Binding handlers extracted to reduce main hook size
const useBindingHandlers = (
  setBindingState: React.Dispatch<React.SetStateAction<BindingState>>,
) => {
  const handleBindingModeChange = (inputName: string, mode: 'event' | 'literal') => {
    setBindingState((prev) => updateBindingMode(prev, inputName, mode));
  };

  const handleBindingValueChange = (inputName: string, value: string) => {
    setBindingState((prev) => updateBindingValue(prev, inputName, value));
  };

  const bindingQuickInsert = (inputName: string, value: string) => {
    setBindingState((prev) => quickInsertBinding(prev, inputName, value));
  };

  return { handleBindingModeChange, handleBindingValueChange, bindingQuickInsert } as const;
};

// Trigger actions extracted
const useTriggerActions = (
  isDraft: boolean,
  subscription: NonNullable<WorkflowNodeData['triggerMeta']>['subscription'],
  handlers: NonNullable<WorkflowNodeData['triggerMeta']>['handlers'],
) => {
  const [isToggling, setIsToggling] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const handleToggle = async (nextEnabled: boolean) => {
    if (isDraft || !subscription) return;
    setIsToggling(true);
    try {
      await toggleTrigger(subscription, nextEnabled, handlers?.toggle);
    } finally {
      setIsToggling(false);
    }
  };

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      await deleteTrigger(subscription, handlers);
    } finally {
      setIsDeleting(false);
    }
  };

  return { isToggling, setIsToggling, isDeleting, setIsDeleting, handleToggle, handleDelete } as const;
};

export const useBaseTrigger = ({
  subscription,
  handlers,
}: {
  subscription: NonNullable<WorkflowNodeData['triggerMeta']>['subscription'];
  handlers: NonNullable<WorkflowNodeData['triggerMeta']>['handlers'];
}) => {
  const isDraft = !subscription;

  const scopedWorkflowInputs = useMemo<Record<string, never>>(() => ({}), []);

  const [bindingState, setBindingState] = useState<BindingState>(() =>
    subscription
      ? deriveBindingStateFromSubscription(scopedWorkflowInputs, subscription)
      : buildDefaultBindingState(scopedWorkflowInputs),
  );
  const [isExpanded, setIsExpanded] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (subscription) {
      setBindingState(deriveBindingStateFromSubscription(scopedWorkflowInputs, subscription));
    } else {
      setBindingState(buildDefaultBindingState(scopedWorkflowInputs));
    }
  }, [subscription, scopedWorkflowInputs]);

  const bindingHandlers = useBindingHandlers(setBindingState);
  const triggerActions = useTriggerActions(isDraft, subscription, handlers);

  const handleCopy = (value: string | null | undefined, label: string) => {
    copyToClipboard(value, label);
  };

  return {
    bindingState,
    setBindingState,
    isExpanded,
    setIsExpanded,
    isSaving,
    setIsSaving,
    ...triggerActions,
    isDraft,
    handleCopy,
    ...bindingHandlers,
    buildBindingsPayload,
  } as const;
};
