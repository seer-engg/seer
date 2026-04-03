import { useRef } from 'react';
import { Plus, Trash2, GripVertical } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import type { HitlInputField, HitlInputOption, HitlTableColumn, TemplateAutocompleteControls } from '../types';
import { VariableAutocompleteDropdown } from '../widgets/VariableAutocompleteDropdown';

interface HITLInputsEditorProps {
  inputs: HitlInputField[];
  onChange: (inputs: HitlInputField[]) => void;
  templateAutocomplete: TemplateAutocompleteControls;
}

const INPUT_TYPES = [
  { value: 'single_choice', label: 'Single Choice' },
  { value: 'multi_choice', label: 'Multiple Choice' },
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'boolean', label: 'Yes/No' },
  { value: 'table', label: 'Table (Batch)' },
] as const;

function generateInputId(): string {
  return `input_${Date.now().toString(36)}`;
}

interface OptionsEditorProps {
  inputIndex: number;
  options: HitlInputOption[];
  onChange: (options: HitlInputOption[]) => void;
  templateAutocomplete: TemplateAutocompleteControls;
  labelInputRefs: React.MutableRefObject<Map<string, HTMLInputElement>>;
}

/**
 * Editor for a single input field's options (for choice types)
 */
function OptionsEditor({
  inputIndex,
  options,
  onChange,
  templateAutocomplete,
  labelInputRefs,
}: OptionsEditorProps) {
  const addOption = () => onChange([...options, { value: '', label: '' }]);
  const updateOption = (index: number, updates: Partial<HitlInputOption>) => {
    const newOptions = [...options];
    newOptions[index] = { ...newOptions[index], ...updates };
    onChange(newOptions);
  };
  const removeOption = (index: number) => {
    labelInputRefs.current.delete(`option-${inputIndex}-${index}`);
    onChange(options.filter((_, i) => i !== index));
  };

  const handleLabelChange = (optIndex: number, newValue: string) => {
    const refKey = `option-${inputIndex}-${optIndex}`;
    const input = labelInputRefs.current.get(refKey);
    if (input) {
      const cursorPos = input.selectionStart || 0;
      templateAutocomplete.checkForAutocomplete(newValue, cursorPos, {
        inputId: refKey,
        ref: { current: input },
        value: newValue,
        onChange: (v: string) => updateOption(optIndex, { label: v }),
      });
    }
    updateOption(optIndex, { label: newValue });
  };

  const handleLabelVariableSelect = (optIndex: number, variable: string) => {
    const refKey = `option-${inputIndex}-${optIndex}`;
    const input = labelInputRefs.current.get(refKey);
    if (input) {
      templateAutocomplete.insertVariable(variable, {
        inputId: refKey,
        ref: { current: input },
        value: options[optIndex].label,
        onChange: (v: string) => updateOption(optIndex, { label: v }),
      });
    }
  };

  const isActiveField = (optIndex: number) =>
    templateAutocomplete.autocompleteContext?.inputId === `option-${inputIndex}-${optIndex}`;

  return (
    <div className="ml-4 space-y-2 border-l-2 border-muted pl-3">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">Options</Label>
        <Button type="button" variant="ghost" size="sm" onClick={addOption} className="h-6 px-2 text-xs">
          <Plus className="w-3 h-3 mr-1" />
          Add Option
        </Button>
      </div>
      {options.map((option, optIndex) => (
        <div key={optIndex} className="flex gap-2 items-center">
          <Input
            placeholder="Value"
            value={option.value}
            onChange={(e) => updateOption(optIndex, { value: e.target.value })}
            className="h-7 text-xs flex-1"
          />
          <div className="relative flex-1">
            <Input
              ref={(el) => { if (el) labelInputRefs.current.set(`option-${inputIndex}-${optIndex}`, el); }}
              placeholder="Label (supports ${variables})"
              value={option.label}
              onChange={(e) => handleLabelChange(optIndex, e.target.value)}
              onKeyDown={(e) => templateAutocomplete.handleKeyDown(e)}
              className="h-7 text-xs"
            />
            {isActiveField(optIndex) && templateAutocomplete.showAutocomplete && (
              <VariableAutocompleteDropdown
                visible={true}
                items={templateAutocomplete.currentLevelItems}
                selectedIndex={templateAutocomplete.selectedIndex}
                currentPath={templateAutocomplete.currentPath}
                onSelect={(variable) => handleLabelVariableSelect(optIndex, variable)}
                onDrillInto={templateAutocomplete.drillInto}
                onNavigateTo={templateAutocomplete.navigateTo}
              />
            )}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => removeOption(optIndex)}
            className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      ))}
      {options.length === 0 && <p className="text-xs text-muted-foreground italic">Add at least one option</p>}
    </div>
  );
}

interface TemplateInputProps {
  refKey: string;
  placeholder: string;
  value: string;
  inputRefs: React.MutableRefObject<Map<string, HTMLInputElement>>;
  templateAutocomplete: TemplateAutocompleteControls;
  onChange: (value: string) => void;
  onVariableSelect: (variable: string) => void;
  className?: string;
}

function TemplateInput({
  refKey,
  placeholder,
  value,
  inputRefs,
  templateAutocomplete,
  onChange,
  onVariableSelect,
  className,
}: TemplateInputProps) {
  const isActive = templateAutocomplete.autocompleteContext?.inputId === refKey;

  const handleChange = (newValue: string) => {
    const inputEl = inputRefs.current.get(refKey);
    if (inputEl) {
      const cursorPos = inputEl.selectionStart || 0;
      templateAutocomplete.checkForAutocomplete(newValue, cursorPos, {
        inputId: refKey,
        ref: { current: inputEl },
        value: newValue,
        onChange,
      });
    }
    onChange(newValue);
  };

  return (
    <div className="relative">
      <Input
        ref={(el) => { if (el) inputRefs.current.set(refKey, el); }}
        placeholder={placeholder}
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        onKeyDown={(e) => templateAutocomplete.handleKeyDown(e)}
        className={className}
      />
      {isActive && templateAutocomplete.showAutocomplete && (
        <VariableAutocompleteDropdown
          visible={true}
          items={templateAutocomplete.currentLevelItems}
          selectedIndex={templateAutocomplete.selectedIndex}
          currentPath={templateAutocomplete.currentPath}
          onSelect={onVariableSelect}
          onDrillInto={templateAutocomplete.drillInto}
          onNavigateTo={templateAutocomplete.navigateTo}
        />
      )}
    </div>
  );
}

interface InputFieldMetaRowProps {
  id: string;
  required: boolean | undefined;
  onUpdate: (updates: Partial<HitlInputField>) => void;
}

function InputFieldMetaRow({ id, required, onUpdate }: InputFieldMetaRowProps) {
  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Checkbox
          id={`required-${id}`}
          checked={required ?? false}
          onCheckedChange={(checked) => onUpdate({ required: checked === true })}
        />
        <Label htmlFor={`required-${id}`} className="text-xs font-normal cursor-pointer">
          Required
        </Label>
      </div>
      <Input
        placeholder="Field ID (auto-generated)"
        value={id}
        onChange={(e) => onUpdate({ id: e.target.value })}
        className="h-7 text-xs w-32 font-mono"
      />
    </div>
  );
}

interface QuestionTypeRowProps {
  questionRefKey: string;
  question: string;
  inputType: HitlInputField['input_type'];
  options: HitlInputField['options'];
  inputRefs: React.MutableRefObject<Map<string, HTMLInputElement>>;
  templateAutocomplete: TemplateAutocompleteControls;
  onQuestionChange: (v: string) => void;
  onQuestionVariableSelect: (v: string) => void;
  onTypeChange: (updates: Partial<HitlInputField>) => void;
}

function QuestionTypeRow({
  questionRefKey, question, inputType, options, inputRefs,
  templateAutocomplete, onQuestionChange, onQuestionVariableSelect, onTypeChange,
}: QuestionTypeRowProps) {
  return (
    <div className="flex gap-2">
      <TemplateInput
        refKey={questionRefKey}
        placeholder="Question (e.g., 'Do you approve ${order.total}?')"
        value={question}
        inputRefs={inputRefs}
        templateAutocomplete={templateAutocomplete}
        onChange={onQuestionChange}
        onVariableSelect={onQuestionVariableSelect}
        className="h-8 text-sm flex-1"
      />
      <Select
        value={inputType}
        onValueChange={(value) =>
          onTypeChange({
            input_type: value as HitlInputField['input_type'],
            options: ['single_choice', 'multi_choice'].includes(value) ? options || [] : undefined,
          })
        }
      >
        <SelectTrigger className="h-8 w-36">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {INPUT_TYPES.map((type) => (
            <SelectItem key={type.value} value={type.value}>
              {type.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

interface TableEditorProps {
  input: HitlInputField;
  inputIndex: number;
  onUpdate: (updates: Partial<HitlInputField>) => void;
  inputRefs: React.MutableRefObject<Map<string, HTMLInputElement>>;
  templateAutocomplete: TemplateAutocompleteControls;
  makeRefKey: (field: string) => string;
  makeVariableSelectHandler: (field: 'question' | 'placeholder' | 'default_value') => (variable: string) => void;
}

function TableEditor({ input, onUpdate, inputRefs, templateAutocomplete, makeRefKey, makeVariableSelectHandler }: TableEditorProps) {
  return (
    <div className="ml-4 space-y-3 border-l-2 border-muted pl-3">
      <div>
        <Label className="text-xs text-muted-foreground">Row Data Expression</Label>
        <TemplateInput
          refKey={makeRefKey('row_data_expression')}
          placeholder="e.g. ${fetch_items.body.results}"
          value={input.row_data_expression || ''}
          inputRefs={inputRefs}
          templateAutocomplete={templateAutocomplete}
          onChange={(v) => onUpdate({ row_data_expression: v })}
          onVariableSelect={makeVariableSelectHandler('question')}
          className="h-7 text-xs mt-1"
        />
      </div>
      <TableDisplayFieldsEditor input={input} onUpdate={onUpdate} />
      <TableColumnsEditor input={input} onUpdate={onUpdate} />
    </div>
  );
}

interface TableDisplayFieldsEditorProps {
  input: HitlInputField;
  onUpdate: (updates: Partial<HitlInputField>) => void;
}

function TableDisplayFieldsEditor({ input, onUpdate }: TableDisplayFieldsEditorProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <Label className="text-xs text-muted-foreground">Display Columns (read-only)</Label>
        <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs"
          onClick={() => onUpdate({ row_display_fields: [...(input.row_display_fields || []), { label: '', value: '' }] })}
        >
          <Plus className="w-3 h-3 mr-1" /> Add
        </Button>
      </div>
      {(input.row_display_fields || []).map((df, dfIdx) => (
        <div key={dfIdx} className="flex gap-2 items-center">
          <Input placeholder="Label" value={df.label} className="h-7 text-xs flex-1"
            onChange={(e) => {
              const updated = [...(input.row_display_fields || [])];
              updated[dfIdx] = { ...updated[dfIdx], label: e.target.value };
              onUpdate({ row_display_fields: updated });
            }}
          />
          <Input placeholder="Value expr (e.g. ${row.title})" value={df.value} className="h-7 text-xs flex-1"
            onChange={(e) => {
              const updated = [...(input.row_display_fields || [])];
              updated[dfIdx] = { ...updated[dfIdx], value: e.target.value };
              onUpdate({ row_display_fields: updated });
            }}
          />
          <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
            onClick={() => {
              const updated = (input.row_display_fields || []).filter((_, i) => i !== dfIdx);
              onUpdate({ row_display_fields: updated });
            }}
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}

interface TableColumnsEditorProps {
  input: HitlInputField;
  onUpdate: (updates: Partial<HitlInputField>) => void;
}

function TableColumnsEditor({ input, onUpdate }: TableColumnsEditorProps) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <Label className="text-xs text-muted-foreground">Input Columns</Label>
        <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs"
          onClick={() => onUpdate({ columns: [...(input.columns || []), { id: `col_${Date.now().toString(36)}`, header: '', input_type: 'text' as const, required: true }] })}
        >
          <Plus className="w-3 h-3 mr-1" /> Add Column
        </Button>
      </div>
      {(input.columns || []).map((col, colIdx) => (
        <div key={colIdx} className="flex gap-2 items-center mb-2">
          <Input placeholder="Column ID" value={col.id} className="h-7 text-xs w-24 font-mono"
            onChange={(e) => {
              const updated = [...(input.columns || [])];
              updated[colIdx] = { ...updated[colIdx], id: e.target.value };
              onUpdate({ columns: updated });
            }}
          />
          <Input placeholder="Header" value={col.header} className="h-7 text-xs flex-1"
            onChange={(e) => {
              const updated = [...(input.columns || [])];
              updated[colIdx] = { ...updated[colIdx], header: e.target.value };
              onUpdate({ columns: updated });
            }}
          />
          <Select value={col.input_type}
            onValueChange={(v) => {
              const updated = [...(input.columns || [])];
              updated[colIdx] = { ...updated[colIdx], input_type: v as HitlTableColumn['input_type'], options: ['single_choice', 'multi_choice'].includes(v) ? col.options || [] : undefined };
              onUpdate({ columns: updated });
            }}
          >
            <SelectTrigger className="h-7 w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INPUT_TYPES.filter(t => t.value !== 'table').map((type) => (
                <SelectItem key={type.value} value={type.value}>{type.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button type="button" variant="ghost" size="sm" className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
            onClick={() => {
              const updated = (input.columns || []).filter((_, i) => i !== colIdx);
              onUpdate({ columns: updated });
            }}
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      ))}
    </div>
  );
}

interface InputFieldRowProps {
  input: HitlInputField;
  inputIndex: number;
  onUpdate: (updates: Partial<HitlInputField>) => void;
  onRemove: () => void;
  templateAutocomplete: TemplateAutocompleteControls;
  inputRefs: React.MutableRefObject<Map<string, HTMLInputElement>>;
}

function InputFieldRow({ input, inputIndex, onUpdate, onRemove, templateAutocomplete, inputRefs }: InputFieldRowProps) {
  const isChoiceType = ['single_choice', 'multi_choice'].includes(input.input_type);
  const isTextType = input.input_type === 'text';
  const makeRefKey = (field: string) => `input-${inputIndex}-${field}`;

  const makeVariableSelectHandler = (field: 'question' | 'placeholder' | 'default_value') => (variable: string) => {
    const refKey = makeRefKey(field);
    const inputEl = inputRefs.current.get(refKey);
    const currentValue = field === 'default_value' ? String(input.default_value || '') : (input[field] || '');
    if (inputEl) {
      templateAutocomplete.insertVariable(variable, {
        inputId: refKey,
        ref: { current: inputEl },
        value: currentValue,
        onChange: (v: string) => onUpdate({ [field]: v }),
      });
    }
  };

  return (
    <div className="p-3 bg-muted/50 rounded-md space-y-3">
      <div className="flex gap-2 items-start">
        <GripVertical className="w-4 h-4 text-muted-foreground mt-2 cursor-grab" />
        <div className="flex-1 space-y-2">
          <QuestionTypeRow
            questionRefKey={makeRefKey('question')}
            question={input.question}
            inputType={input.input_type}
            options={input.options}
            inputRefs={inputRefs}
            templateAutocomplete={templateAutocomplete}
            onQuestionChange={(v) => onUpdate({ question: v })}
            onQuestionVariableSelect={makeVariableSelectHandler('question')}
            onTypeChange={onUpdate}
          />
          {(isTextType || input.input_type === 'number') && (
            <TemplateInput
              refKey={makeRefKey('placeholder')}
              placeholder="Placeholder text (supports ${variables})"
              value={input.placeholder || ''}
              inputRefs={inputRefs}
              templateAutocomplete={templateAutocomplete}
              onChange={(v) => onUpdate({ placeholder: v })}
              onVariableSelect={makeVariableSelectHandler('placeholder')}
              className="h-7 text-xs"
            />
          )}
          {isTextType && (
            <TemplateInput
              refKey={makeRefKey('default_value')}
              placeholder="Default value (supports ${variables})"
              value={String(input.default_value || '')}
              inputRefs={inputRefs}
              templateAutocomplete={templateAutocomplete}
              onChange={(v) => onUpdate({ default_value: v })}
              onVariableSelect={makeVariableSelectHandler('default_value')}
              className="h-7 text-xs"
            />
          )}
          <InputFieldMetaRow id={input.id} required={input.required} onUpdate={onUpdate} />
          {isChoiceType && (
            <OptionsEditor
              inputIndex={inputIndex}
              options={input.options || []}
              onChange={(options) => onUpdate({ options })}
              templateAutocomplete={templateAutocomplete}
              labelInputRefs={inputRefs}
            />
          )}
          {input.input_type === 'table' && (
            <TableEditor
              input={input}
              inputIndex={inputIndex}
              onUpdate={onUpdate}
              inputRefs={inputRefs}
              templateAutocomplete={templateAutocomplete}
              makeRefKey={makeRefKey}
              makeVariableSelectHandler={makeVariableSelectHandler}
            />
          )}
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
    </div>
  );
}

/**
 * Editor for HITL input fields.
 * Input fields collect user responses during workflow interruption.
 * Supports template variables (${variable}) in question, placeholder, default value, and option labels.
 */
export function HITLInputsEditor({ inputs, onChange, templateAutocomplete }: HITLInputsEditorProps) {
  const inputRefs = useRef<Map<string, HTMLInputElement>>(new Map());

  const addInput = () => {
    const newInput: HitlInputField = { id: generateInputId(), question: '', input_type: 'text', required: true };
    onChange([...inputs, newInput]);
  };

  const updateInput = (index: number, updates: Partial<HitlInputField>) => {
    const newInputs = [...inputs];
    newInputs[index] = { ...newInputs[index], ...updates };
    onChange(newInputs);
  };

  const removeInput = (index: number) => {
    // Clean up refs for removed input
    const keysToDelete = Array.from(inputRefs.current.keys()).filter((key) => key.startsWith(`input-${index}-`));
    keysToDelete.forEach((key) => inputRefs.current.delete(key));
    onChange(inputs.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Label className="text-sm font-medium">Input Fields</Label>
        <Button type="button" variant="outline" size="sm" onClick={addInput} className="h-7 px-2">
          <Plus className="w-3 h-3 mr-1" />
          Add Field
        </Button>
      </div>
      <p className="text-xs text-muted-foreground">
        Define questions and input types for collecting human responses. Use {'${variable}'} syntax to reference node
        outputs.
      </p>

      {inputs.length === 0 ? (
        <div className="text-xs text-muted-foreground py-2 text-center border border-dashed rounded-md">
          No input fields configured
        </div>
      ) : (
        <div className="space-y-3">
          {inputs.map((input, index) => (
            <InputFieldRow
              key={input.id}
              input={input}
              inputIndex={index}
              onUpdate={(updates) => updateInput(index, updates)}
              onRemove={() => removeInput(index)}
              templateAutocomplete={templateAutocomplete}
              inputRefs={inputRefs}
            />
          ))}
        </div>
      )}
    </div>
  );
}
