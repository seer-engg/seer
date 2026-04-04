import { WorkflowInputDialog } from '@/components/workflows/dialogs/WorkflowInputDialog';
import { KeymapDialog } from '@/components/general/KeymapDialog';
import { RunOptionsDialog } from '@/components/workflows/dialogs/RunOptionsDialog';
import { TriggerEventPicker } from '@/components/workflows/dialogs/TriggerEventPicker';
import type { TriggerWithConnection } from '@/components/workflows/dialogs/RunOptionsDialog';

interface WorkflowDialogsProps {
  isInputDialogOpen: boolean;
  isKeymapOpen: boolean;
  inputFields: Array<{
    id: string;
    label: string;
    type: string;
    required: boolean;
    variableName: string;
  }>;
  inputData: Record<string, unknown>;
  setInputDialogOpen: (open: boolean) => void;
  setInputData: (data: Record<string, unknown>) => void;
  setKeymapOpen: (open: boolean) => void;
  onExecuteWithInput: () => void;
  // Run options dialog props
  isRunOptionsDialogOpen: boolean;
  setRunOptionsDialogOpen: (open: boolean) => void;
  triggersWithConnection: TriggerWithConnection[];
  onSelectTriggerForRealEvent: (trigger: TriggerWithConnection) => void;
  onConnectProvider?: (provider: string) => void;
  /** Handler for instant triggers (e.g., cron) - triggers workflow immediately */
  onTriggerNow?: (trigger: TriggerWithConnection) => void;
  /** Whether an instant trigger is currently being executed */
  isTriggering?: boolean;
  // Trigger event picker props
  isTriggerEventPickerOpen: boolean;
  setTriggerEventPickerOpen: (open: boolean) => void;
  selectedTriggerForPicker: {
    triggerId: string;
    triggerKey: string;
    provider: string;
    connectionId?: number;
    subscriptionId?: number;
    filterParams?: Record<string, unknown>;
  } | null;
  onTriggerEventSelect: (event: { id: string; envelope: Record<string, unknown> }) => void;
}

export function WorkflowDialogs({
  isInputDialogOpen,
  isKeymapOpen,
  inputFields,
  inputData,
  setInputDialogOpen,
  setInputData,
  setKeymapOpen,
  onExecuteWithInput,
  // Run options
  isRunOptionsDialogOpen,
  setRunOptionsDialogOpen,
  triggersWithConnection,
  onSelectTriggerForRealEvent,
  onConnectProvider,
  onTriggerNow,
  isTriggering,
  // Trigger event picker
  isTriggerEventPickerOpen,
  setTriggerEventPickerOpen,
  selectedTriggerForPicker,
  onTriggerEventSelect,
}: WorkflowDialogsProps) {
  return (
    <>
      <WorkflowInputDialog
        open={isInputDialogOpen}
        onOpenChange={setInputDialogOpen}
        inputFields={inputFields}
        inputData={inputData}
        onInputDataChange={setInputData}
        onExecute={onExecuteWithInput}
      />

      <KeymapDialog open={isKeymapOpen} onOpenChange={setKeymapOpen} />

      <RunOptionsDialog
        open={isRunOptionsDialogOpen}
        onOpenChange={setRunOptionsDialogOpen}
        triggers={triggersWithConnection}
        onSelectTrigger={onSelectTriggerForRealEvent}
        onConnectProvider={onConnectProvider}
        onTriggerNow={onTriggerNow}
        isTriggering={isTriggering}
      />

      {selectedTriggerForPicker && (
        <TriggerEventPicker
          open={isTriggerEventPickerOpen}
          onOpenChange={setTriggerEventPickerOpen}
          triggerId={selectedTriggerForPicker.triggerId}
          triggerKey={selectedTriggerForPicker.triggerKey}
          provider={selectedTriggerForPicker.provider}
          providerConnectionId={selectedTriggerForPicker.connectionId}
          subscriptionId={selectedTriggerForPicker.subscriptionId}
          filterParams={selectedTriggerForPicker.filterParams}
          onEventSelect={onTriggerEventSelect}
        />
      )}
    </>
  );
}
