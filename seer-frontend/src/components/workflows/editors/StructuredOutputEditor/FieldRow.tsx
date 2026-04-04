/**
 * Recursive field row for the StructuredOutputEditor visual mode.
 */

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Plus, Trash2 } from 'lucide-react';
import { PYDANTIC_TYPES, createEmptyField } from './utils';
import type { FieldDefinition } from './types';

const MAX_DEPTH = 3;

// ─── FieldRowHeader ───────────────────────────────────────────────────────────

interface FieldRowHeaderProps {
  field: FieldDefinition;
  depth: number;
  index: number;
  canRemove: boolean;
  onUpdate: (index: number, updates: Partial<FieldDefinition>) => void;
  onRemove: (index: number) => void;
}

function FieldRowHeader({ field, depth, index, canRemove, onUpdate, onRemove }: FieldRowHeaderProps) {
  const availableTypes = depth >= MAX_DEPTH - 1
    ? PYDANTIC_TYPES.filter((t) => t.value !== 'dict')
    : PYDANTIC_TYPES;

  return (
    <div className="space-y-3">
      <div className="flex gap-2 items-start">
        <div className="flex-1 space-y-1">
          <Label className="text-xs font-medium text-muted-foreground">Name</Label>
          <Input
            value={field.name}
            onChange={(e) => onUpdate(index, { name: e.target.value })}
            placeholder="field_name"
            className="h-8 text-sm"
          />
        </div>
        <div className="w-36 space-y-1">
          <Label className="text-xs font-medium text-muted-foreground">Type</Label>
          <Select
            value={field.type}
            onValueChange={(value) => {
              const updates: Partial<FieldDefinition> = { type: value };
              if (value !== 'dict') updates.children = undefined;
              if (value !== 'list') { updates.itemType = undefined; updates.itemChildren = undefined; }
              onUpdate(index, updates);
            }}
          >
            <SelectTrigger className="h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {availableTypes.map((t) => (
                <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="pt-6">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-muted-foreground hover:text-destructive"
            onClick={() => onRemove(index)}
            disabled={!canRemove}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>
      <Textarea
        value={field.description ?? ''}
        onChange={(e) => onUpdate(index, { description: e.target.value })}
        placeholder="Description (optional)..."
        className="min-h-[56px] text-sm resize-none"
      />
    </div>
  );
}

// ─── ChildFieldsList ──────────────────────────────────────────────────────────

interface ChildFieldsListProps {
  fields: FieldDefinition[];
  depth: number;
  label: string;
  onFieldsChange: (updated: FieldDefinition[]) => void;
}

function ChildFieldsList({ fields, depth, label, onFieldsChange }: ChildFieldsListProps) {
  const handleUpdate = (i: number, updates: Partial<FieldDefinition>) => {
    const next = [...fields];
    next[i] = { ...next[i], ...updates };
    onFieldsChange(next);
  };

  const handleRemove = (i: number) => {
    const next = fields.filter((_, idx) => idx !== i);
    onFieldsChange(next.length > 0 ? next : []);
  };

  return (
    <div className="border-l-2 border-muted ml-3 pl-3 mt-3 space-y-2">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      {fields.map((child, i) => (
        <FieldRow
          key={child.id}
          field={child}
          index={i}
          depth={depth}
          canRemove={fields.length > 1}
          onUpdate={handleUpdate}
          onRemove={handleRemove}
        />
      ))}
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-7 text-xs text-muted-foreground"
        onClick={() => onFieldsChange([...fields, createEmptyField()])}
      >
        <Plus className="w-3 h-3 mr-1" />
        Add field
      </Button>
    </div>
  );
}

// ─── ListItemConfig ───────────────────────────────────────────────────────────

interface ListItemConfigProps {
  field: FieldDefinition;
  index: number;
  depth: number;
  onUpdate: (index: number, updates: Partial<FieldDefinition>) => void;
}

function ListItemConfig({ field, index, depth, onUpdate }: ListItemConfigProps) {
  const scalarTypes = PYDANTIC_TYPES.filter((t) => !['list', 'dict'].includes(t.value));
  const itemTypes = depth < MAX_DEPTH - 1
    ? [{ value: 'object', label: 'Object' }, ...scalarTypes]
    : scalarTypes;

  return (
    <div className="mt-2 space-y-2">
      <div className="flex items-center gap-2">
        <Label className="text-xs font-medium text-muted-foreground whitespace-nowrap">Item type</Label>
        <Select
          value={field.itemType ?? 'str'}
          onValueChange={(value) => {
            const updates: Partial<FieldDefinition> = { itemType: value };
            if (value !== 'object') updates.itemChildren = undefined;
            onUpdate(index, updates);
          }}
        >
          <SelectTrigger className="h-8 text-sm w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {itemTypes.map((t) => (
              <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {field.itemType === 'object' && (
        <ChildFieldsList
          fields={field.itemChildren?.length ? field.itemChildren : [createEmptyField()]}
          depth={depth + 1}
          label="Item properties"
          onFieldsChange={(updated) => onUpdate(index, { itemChildren: updated })}
        />
      )}
    </div>
  );
}

// ─── FieldRow (orchestrator) ──────────────────────────────────────────────────

export interface FieldRowProps {
  field: FieldDefinition;
  index: number;
  depth: number;
  onUpdate: (index: number, updates: Partial<FieldDefinition>) => void;
  onRemove: (index: number) => void;
  canRemove: boolean;
}

export function FieldRow({ field, index, depth, onUpdate, onRemove, canRemove }: FieldRowProps) {
  return (
    <div className="rounded-lg border p-3 space-y-1 bg-muted/30">
      <FieldRowHeader
        field={field}
        depth={depth}
        index={index}
        canRemove={canRemove}
        onUpdate={onUpdate}
        onRemove={onRemove}
      />

      {field.type === 'dict' && (
        <ChildFieldsList
          fields={field.children?.length ? field.children : [createEmptyField()]}
          depth={depth + 1}
          label="Properties"
          onFieldsChange={(updated) => onUpdate(index, { children: updated })}
        />
      )}

      {field.type === 'list' && (
        <ListItemConfig
          field={field}
          index={index}
          depth={depth}
          onUpdate={onUpdate}
        />
      )}
    </div>
  );
}
