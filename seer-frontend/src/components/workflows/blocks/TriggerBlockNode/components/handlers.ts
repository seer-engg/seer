import { toast } from '@/components/ui/sonner';
import type { WorkflowNodeData } from '../../../types';
import type { BindingState } from '../../../triggers/utils';

const fallbackCopyToClipboard = (value: string) => {
  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.top = '-9999px';
  textarea.style.left = '-9999px';

  document.body.appendChild(textarea);
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

  const copied = document.execCommand('copy');
  document.body.removeChild(textarea);

  if (!copied) {
    throw new Error('execCommand copy failed');
  }
};

export const copyToClipboard = async (value: string | null | undefined, label: string) => {
  if (!value) {
    toast.error(`No ${label.toLowerCase()} available`);
    return;
  }
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(value);
    } else {
      fallbackCopyToClipboard(value);
    }
    toast.success(`${label} copied`);
  } catch (error) {
    console.error('Clipboard error', error);
    toast.error('Unable to copy');
  }
};

export const toggleTrigger = async (
  subscription: WorkflowNodeData['triggerMeta']['subscription'],
  nextEnabled: boolean,
  toggleHandler: ((triggerId: string, enabled: boolean) => Promise<void>) | undefined,
) => {
  if (!toggleHandler || !subscription) {
    return;
  }
  try {
    await toggleHandler(subscription.id, nextEnabled);
    toast.success(`Trigger ${nextEnabled ? 'enabled' : 'disabled'}`);
  } catch (error) {
    console.error('Failed to toggle trigger', error);
    toast.error('Unable to update trigger');
    throw error;
  }
};

export const deleteTrigger = async (
  subscription: WorkflowNodeData['triggerMeta']['subscription'],
  handlers: NonNullable<WorkflowNodeData['triggerMeta']['handlers']>,
) => {
  if (!handlers?.delete || !subscription) {
    console.warn('Cannot delete trigger: missing handlers or subscription');
    return;
  }
  if (!confirm('Delete this trigger?')) {
    return;
  }
  try {
    await handlers.delete(subscription.id);
    toast.success('Trigger removed');
  } catch (error) {
    console.error('Failed to delete trigger', error);
    toast.error('Unable to delete trigger');
    throw error;
  }
};

export const updateBindingMode = (
  bindingState: BindingState,
  inputName: string,
  mode: 'event' | 'literal',
): BindingState => {
  return {
    ...bindingState,
    [inputName]: {
      mode,
      value:
        mode === 'event'
          ? bindingState[inputName]?.mode === 'event'
            ? bindingState[inputName]?.value ?? `data.${inputName}`
            : `data.${inputName}`
          : bindingState[inputName]?.mode === 'literal'
            ? bindingState[inputName]?.value ?? ''
            : '',
    },
  };
};

export const updateBindingValue = (
  bindingState: BindingState,
  inputName: string,
  value: string,
): BindingState => {
  return {
    ...bindingState,
    [inputName]: {
      mode: bindingState[inputName]?.mode ?? 'event',
      value,
    },
  };
};

export const quickInsertBinding = (
  bindingState: BindingState,
  inputName: string,
  value: string,
): BindingState => {
  return {
    ...bindingState,
    [inputName]: {
      mode: 'event',
      value,
    },
  };
};
