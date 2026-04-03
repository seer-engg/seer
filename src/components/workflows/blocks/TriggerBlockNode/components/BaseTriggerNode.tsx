import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Info, Loader2, Trash2 } from 'lucide-react';

import type { InputDef } from '@/types/workflow-spec';
import { INPUT_TYPE_OPTIONS } from './inputConstants';

export interface QuickOption {
  label: string;
  path: string;
}
// Note: Hook logic moved to components/useBaseTrigger.ts to satisfy
// react-refresh/only-export-components and reduce function lengths.

export interface WorkflowInputManagerProps {
  canManageInputs: boolean;
  showInputsEditor: boolean;
  setShowInputsEditor: (show: boolean | ((prev: boolean) => boolean)) => void;
  workflowInputEntries: Array<[string, InputDef]>;
  isSavingWorkflowInput: boolean;
  inputDraft: {
    name: string;
    type: InputDef['type'];
    description: string;
    required: boolean;
  };
  setInputDraft: React.Dispatch<
    React.SetStateAction<{
      name: string;
      type: InputDef['type'];
      description: string;
      required: boolean;
    }>
  >;
  handleCreateWorkflowInput: () => void;
  handleRemoveWorkflowInput: (inputName: string) => void;
}

export const WorkflowInputManager: React.FC<WorkflowInputManagerProps> = ({
  canManageInputs,
  showInputsEditor,
  setShowInputsEditor,
  workflowInputEntries,
  isSavingWorkflowInput,
  inputDraft,
  setInputDraft,
  handleCreateWorkflowInput,
  handleRemoveWorkflowInput,
}) => {
  if (!canManageInputs) return null;

  return (
    <div className="rounded-lg border border-dashed bg-muted/40 p-4 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium">Workflow inputs</p>
          <p className="text-xs text-muted-foreground">
            Add inputs for this trigger to populate before configuring bindings.
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => setShowInputsEditor((p) => !p)}>
          {showInputsEditor ? 'Hide form' : 'Add input'}
        </Button>
      </div>

      {workflowInputEntries.length === 0 ? (
        <p className="text-xs text-muted-foreground">No workflow inputs yet. Add one to bind incoming event data.</p>
      ) : (
        <WorkflowInputList
          entries={workflowInputEntries}
          isSaving={isSavingWorkflowInput}
          onRemove={handleRemoveWorkflowInput}
        />
      )}

      {showInputsEditor && (
        <WorkflowInputForm
          inputDraft={inputDraft}
          setInputDraft={setInputDraft}
          isSaving={isSavingWorkflowInput}
          onSave={handleCreateWorkflowInput}
        />
      )}
    </div>
  );
};

const WorkflowInputList: React.FC<{
  entries: Array<[string, InputDef]>;
  isSaving: boolean;
  onRemove: (inputName: string) => void;
}> = ({ entries, isSaving, onRemove }) => (
  <div className="space-y-2">
    {entries.map(([inputName, inputDef]) => (
      <div key={inputName} className="flex items-center justify-between gap-3 rounded border bg-background px-3 py-2">
        <div>
          <p className="text-sm font-medium">{inputName}</p>
          <p className="text-xs text-muted-foreground">{inputDef.description || 'No description provided'}</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <Badge variant="secondary">{inputDef.type}</Badge>
          {inputDef.required && <Badge variant="outline">Required</Badge>}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={() => onRemove(inputName)}
            disabled={isSaving}
            title="Remove workflow input"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>
    ))}
  </div>
);

const WorkflowInputForm: React.FC<{
  inputDraft: {
    name: string;
    type: InputDef['type'];
    description: string;
    required: boolean;
  };
  setInputDraft: React.Dispatch<React.SetStateAction<{ name: string; type: InputDef['type']; description: string; required: boolean }>>;
  isSaving: boolean;
  onSave: () => void;
}> = ({ inputDraft, setInputDraft, isSaving, onSave }) => (
  <div className="space-y-3 border-t border-border/50 pt-3">
    <div className="space-y-1">
      <Label className="text-xs uppercase text-muted-foreground">Name</Label>
      <Input
        value={inputDraft.name}
        placeholder="customerEmail"
        onChange={(event) =>
          setInputDraft((prev) => ({
            ...prev,
            name: event.target.value,
          }))
        }
      />
    </div>
    <div className="grid gap-3 sm:grid-cols-2">
      <div className="space-y-1">
        <Label className="text-xs uppercase text-muted-foreground">Type</Label>
        <Select
          value={inputDraft.type}
          onValueChange={(value) =>
            setInputDraft((prev) => ({
              ...prev,
              type: value as InputDef['type'],
            }))
          }
        >
          <SelectTrigger>
            <SelectValue placeholder="Select type" />
          </SelectTrigger>
          <SelectContent>
            {INPUT_TYPE_OPTIONS.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-1">
        <Label className="text-xs uppercase text-muted-foreground">Required</Label>
        <div className="flex items-center gap-2">
          <Switch
            checked={inputDraft.required}
            onCheckedChange={(value) =>
              setInputDraft((prev) => ({
                ...prev,
                required: value,
              }))
            }
          />
          <span className="text-xs text-muted-foreground">{inputDraft.required ? 'Required' : 'Optional'}</span>
        </div>
      </div>
    </div>
    <div className="space-y-1">
      <Label className="text-xs uppercase text-muted-foreground">Description</Label>
      <Input
        value={inputDraft.description}
        placeholder="Shown in manual runs"
        onChange={(event) =>
          setInputDraft((prev) => ({
            ...prev,
            description: event.target.value,
          }))
        }
      />
    </div>
    <Button type="button" size="sm" onClick={onSave} disabled={isSaving}>
      {isSaving ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Saving...
        </>
      ) : (
        'Save input'
      )}
    </Button>
  </div>
);