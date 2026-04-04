import { useState, useRef, useEffect, useCallback } from 'react';
import { Check, X, Pencil } from 'lucide-react';

import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { validateNodeId } from '@/lib/workflow-rename';

export interface NodeIdEditorProps {
  nodeId: string;
  existingNodeIds: string[];
  onRename: (oldId: string, newId: string) => { success: boolean; error?: string };
  disabled?: boolean;
}

interface DisplayModeProps {
  nodeId: string;
  disabled: boolean;
  onEdit: () => void;
}

function DisplayMode({ nodeId, disabled, onEdit }: DisplayModeProps) {
  return (
    <div className="flex items-center gap-2 group">
      <code className="flex-1 text-xs font-mono bg-muted/50 px-2 py-1.5 rounded border border-border truncate">
        {nodeId}
      </code>
      {!disabled && (
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
          onClick={onEdit}
          title="Edit node ID"
        >
          <Pencil className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  );
}

interface EditModeProps {
  value: string;
  error: string | null;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onChange: (value: string) => void;
  onConfirm: () => void;
  onCancel: () => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}

function EditMode({ value, error, inputRef, onChange, onConfirm, onCancel, onKeyDown }: EditModeProps) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1">
        <Input
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          className={cn(
            'h-8 text-xs font-mono flex-1',
            error && 'border-destructive focus-visible:ring-destructive',
          )}
          placeholder="Enter node ID"
        />
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-emerald-600 hover:text-emerald-700 hover:bg-emerald-500/10"
          onClick={onConfirm}
          title="Confirm"
        >
          <Check className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground hover:text-foreground"
          onClick={onCancel}
          title="Cancel"
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>
      {error && <p className="text-[10px] text-destructive pl-0.5">{error}</p>}
    </div>
  );
}

export function NodeIdEditor({ nodeId, existingNodeIds, onRename, disabled = false }: NodeIdEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(nodeId);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setEditValue(nodeId);
    setError(null);
    setIsEditing(false);
  }, [nodeId]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const handleStartEdit = useCallback(() => {
    if (disabled) return;
    setEditValue(nodeId);
    setError(null);
    setIsEditing(true);
  }, [disabled, nodeId]);

  const handleCancel = useCallback(() => {
    setEditValue(nodeId);
    setError(null);
    setIsEditing(false);
  }, [nodeId]);

  const handleConfirm = useCallback(() => {
    const trimmedValue = editValue.trim();
    if (trimmedValue === nodeId) {
      setIsEditing(false);
      return;
    }
    const validation = validateNodeId(trimmedValue, existingNodeIds, nodeId);
    if (!validation.valid) {
      setError(validation.error ?? 'Invalid ID');
      return;
    }
    const result = onRename(nodeId, trimmedValue);
    if (!result.success) {
      setError(result.error ?? 'Rename failed');
      return;
    }
    setError(null);
    setIsEditing(false);
  }, [editValue, nodeId, existingNodeIds, onRename]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleConfirm();
      } else if (e.key === 'Escape') {
        e.preventDefault();
        handleCancel();
      }
    },
    [handleConfirm, handleCancel],
  );

  const handleChange = useCallback((value: string) => {
    setEditValue(value);
    setError(null);
  }, []);

  if (!isEditing) {
    return <DisplayMode nodeId={nodeId} disabled={disabled} onEdit={handleStartEdit} />;
  }

  return (
    <EditMode
      value={editValue}
      error={error}
      inputRef={inputRef}
      onChange={handleChange}
      onConfirm={handleConfirm}
      onCancel={handleCancel}
      onKeyDown={handleKeyDown}
    />
  );
}
