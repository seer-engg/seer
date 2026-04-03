import { useRef } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import type { HitlDisplayItem, TemplateAutocompleteControls } from '../types';
import { VariableAutocompleteDropdown } from '../widgets/VariableAutocompleteDropdown';
import type { VariableTreeItem } from '../helpers/variableTreeUtils';

interface HITLDisplayEditorProps {
  display: HitlDisplayItem[];
  onChange: (display: HitlDisplayItem[]) => void;
  templateAutocomplete: TemplateAutocompleteControls;
}

interface DisplayItemRowProps {
  item: HitlDisplayItem;
  index: number;
  inputRef: (el: HTMLInputElement | null) => void;
  onLabelChange: (label: string) => void;
  onValueChange: (value: string) => void;
  onRemove: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  showAutocomplete: boolean;
  autocompleteItems: VariableTreeItem[];
  autocompleteSelectedIndex: number;
  currentPath: string;
  onAutocompleteSelect: (variable: string) => void;
  onDrillInto: (item: VariableTreeItem) => void;
  onNavigateTo: (path: string) => void;
}

function DisplayItemRow({
  item,
  inputRef,
  onLabelChange,
  onValueChange,
  onRemove,
  onKeyDown,
  showAutocomplete,
  autocompleteItems,
  autocompleteSelectedIndex,
  currentPath,
  onAutocompleteSelect,
  onDrillInto,
  onNavigateTo,
}: DisplayItemRowProps) {
  return (
    <div className="flex gap-2 items-start p-2 bg-muted/50 rounded-md">
      <div className="flex-1 space-y-2">
        <Input
          placeholder="Label (e.g., 'Customer Name')"
          value={item.label}
          onChange={(e) => onLabelChange(e.target.value)}
          className="h-8 text-sm"
        />
        <div className="relative">
          <Input
            ref={inputRef}
            placeholder="Value (e.g., ${customer.name})"
            value={item.value}
            onChange={(e) => onValueChange(e.target.value)}
            onKeyDown={onKeyDown}
            className="h-8 text-sm font-mono"
          />
          {showAutocomplete && (
            <VariableAutocompleteDropdown
              visible={showAutocomplete}
              items={autocompleteItems}
              selectedIndex={autocompleteSelectedIndex}
              currentPath={currentPath}
              onSelect={onAutocompleteSelect}
              onDrillInto={onDrillInto}
              onNavigateTo={onNavigateTo}
            />
          )}
        </div>
      </div>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onRemove}
        className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
      >
        <Trash2 className="w-4 h-4" />
      </Button>
    </div>
  );
}

/**
 * Editor for HITL display items.
 * Display items show workflow data to the human reviewer.
 */
export function HITLDisplayEditor({
  display,
  onChange,
  templateAutocomplete,
}: HITLDisplayEditorProps) {
  const inputRefs = useRef<Map<number, HTMLInputElement>>(new Map());

  const addDisplayItem = () => onChange([...display, { label: '', value: '' }]);

  const updateDisplayItem = (index: number, updates: Partial<HitlDisplayItem>) => {
    const newDisplay = [...display];
    newDisplay[index] = { ...newDisplay[index], ...updates };
    onChange(newDisplay);
  };

  const removeDisplayItem = (index: number) => {
    inputRefs.current.delete(index);
    onChange(display.filter((_, i) => i !== index));
  };

  const handleValueChange = (index: number, newValue: string) => {
    const input = inputRefs.current.get(index);
    if (input) {
      const cursorPos = input.selectionStart || 0;
      templateAutocomplete.checkForAutocomplete(newValue, cursorPos, {
        inputId: `display-${index}-value`,
        ref: { current: input },
        value: newValue,
        onChange: (v: string) => updateDisplayItem(index, { value: v }),
      });
    }
    updateDisplayItem(index, { value: newValue });
  };

  const handleVariableSelect = (index: number, variable: string) => {
    const input = inputRefs.current.get(index);
    if (input) {
      templateAutocomplete.insertVariable(variable, {
        inputId: `display-${index}-value`,
        ref: { current: input },
        value: display[index].value,
        onChange: (v: string) => updateDisplayItem(index, { value: v }),
      });
    }
  };

  const isActiveField = (index: number) =>
    templateAutocomplete.autocompleteContext?.inputId === `display-${index}-value`;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">Display Items</Label>
        <Button type="button" variant="outline" size="sm" onClick={addDisplayItem} className="h-7 px-2">
          <Plus className="w-3 h-3 mr-1" />
          Add
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Show workflow data to the reviewer. Use {'${variable}'} syntax to reference node outputs.
      </p>

      {display.length === 0 ? (
        <div className="text-xs text-muted-foreground py-2 text-center border border-dashed rounded-md">
          No display items configured
        </div>
      ) : (
        <div className="space-y-2">
          {display.map((item, index) => (
            <DisplayItemRow
              key={index}
              item={item}
              index={index}
              inputRef={(el) => { if (el) inputRefs.current.set(index, el); }}
              onLabelChange={(label) => updateDisplayItem(index, { label })}
              onValueChange={(value) => handleValueChange(index, value)}
              onRemove={() => removeDisplayItem(index)}
              onKeyDown={(e) => templateAutocomplete.handleKeyDown(e)}
              showAutocomplete={isActiveField(index) && templateAutocomplete.showAutocomplete}
              autocompleteItems={templateAutocomplete.currentLevelItems}
              autocompleteSelectedIndex={templateAutocomplete.selectedIndex}
              currentPath={templateAutocomplete.currentPath}
              onAutocompleteSelect={(variable) => handleVariableSelect(index, variable)}
              onDrillInto={templateAutocomplete.drillInto}
              onNavigateTo={templateAutocomplete.navigateTo}
            />
          ))}
        </div>
      )}
    </div>
  );
}
