import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from '@/components/ui/alert-dialog';

export interface WorkflowInputField {
  id: string;
  label: string;
  type: string;
  required: boolean;
  variableName: string;
}

export interface WorkflowInputDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  inputFields: WorkflowInputField[];
  inputData: Record<string, unknown>;
  onInputDataChange: (data: Record<string, unknown>) => void;
  onExecute: () => void;
}

export function WorkflowInputDialog({
  open,
  onOpenChange,
  inputFields,
  inputData,
  onInputDataChange,
  onExecute,
}: WorkflowInputDialogProps) {
  return (
    <AlertDialog open={open} onOpenChange={onOpenChange}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Workflow Input</AlertDialogTitle>
          <AlertDialogDescription>
            Please provide input values for this workflow
          </AlertDialogDescription>
        </AlertDialogHeader>
        <div className="space-y-4 py-4">
          {inputFields.map((field) => (
            <div key={field.id} className="space-y-2">
              <Label htmlFor={field.id}>{field.label}</Label>
              <Input
                id={field.id}
                type={field.type}
                value={String(inputData[field.id] || '')}
                onChange={(e) => onInputDataChange({ ...inputData, [field.id]: e.target.value })}
                required={field.required}
                placeholder={`Enter ${field.label.toLowerCase()}`}
              />
            </div>
          ))}
        </div>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction onClick={onExecute}>
            Run Workflow
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
