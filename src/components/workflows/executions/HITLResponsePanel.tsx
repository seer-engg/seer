/**
 * HITLResponsePanel Component
 *
 * Displays the HITL interrupt data and collects user responses.
 * Shows display items (context data) and input fields for the human reviewer.
 */
import { useState, useCallback } from 'react';
import { UserCheck, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { DisplayValue } from './DisplayValue';
import { RichDisplayValue } from './RichDisplayValue';
import { analyzeContent } from '@/lib/content-analysis';
import type { HitlInterruptData, HitlResumePayload } from './types';

/**
 * Renders a single cell input for a table column
 */
function TableCellInput({
  column,
  value,
  onChange,
  rowIndex,
}: {
  column: { id: string; input_type: string; options?: Array<{ value: string; label: string }>; };
  value: unknown;
  onChange: (value: unknown) => void;
  rowIndex: number;
}) {
  switch (column.input_type) {
    case 'single_choice':
      return (
        <select
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-full rounded-md border border-input bg-background px-2 text-sm"
        >
          <option value="">--</option>
          {column.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      );
    case 'boolean':
      return (
        <Checkbox
          id={`table-${rowIndex}-${column.id}`}
          checked={Boolean(value)}
          onCheckedChange={(checked) => onChange(checked === true)}
        />
      );
    case 'number':
      return (
        <Input
          type="number"
          value={value === undefined || value === '' ? '' : Number(value)}
          onChange={(e) => onChange(e.target.value ? Number(e.target.value) : '')}
          className="h-8 text-sm"
        />
      );
    case 'text':
    default:
      return (
        <Input
          value={String(value || '')}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 text-sm"
          placeholder="..."
        />
      );
  }
}

/**
 * Renders a table input with rows from runtime data and per-row input columns
 */
function TableInput({
  input,
  value,
  onChange,
}: {
  input: HitlInterruptData['inputs'][0];
  value: Array<Record<string, unknown>>;
  onChange: (value: Array<Record<string, unknown>>) => void;
}) {
  const rows = input.rows || [];
  const columns = input.columns || [];

  // Get display field labels from first row
  const displayLabels = rows.length > 0 ? Object.keys(rows[0].display || {}) : [];

  const updateCell = (rowIndex: number, colId: string, cellValue: unknown) => {
    const newValue = [...value];
    newValue[rowIndex] = { ...newValue[rowIndex], [colId]: cellValue };
    onChange(newValue);
  };

  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground italic">No rows to display</p>;
  }

  return (
    <div className="overflow-x-auto border rounded-md">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            {displayLabels.map((label) => (
              <th key={label} className="px-3 py-2 text-left font-medium text-muted-foreground">{label}</th>
            ))}
            {columns.map((col) => (
              <th key={col.id} className="px-3 py-2 text-left font-medium">
                {col.header}
                {col.required !== false && <span className="text-destructive ml-1">*</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIdx) => (
            <tr key={rowIdx} className="border-b last:border-0">
              {displayLabels.map((label) => (
                <td key={label} className="px-3 py-2 text-muted-foreground max-w-[200px] truncate">
                  <DisplayValue value={(row.display as Record<string, unknown>)?.[label]} />
                </td>
              ))}
              {columns.map((col) => (
                <td key={col.id} className="px-3 py-2">
                  <TableCellInput
                    column={col}
                    value={value[rowIdx]?.[col.id]}
                    onChange={(v) => updateCell(rowIdx, col.id, v)}
                    rowIndex={rowIdx}
                  />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface HITLResponsePanelProps {
  interruptData: HitlInterruptData;
  onSubmit: (payload: HitlResumePayload) => void;
  isSubmitting?: boolean;
  error?: Error | null;
}

type ResponseValue = string | number | boolean | string[] | Array<Record<string, unknown>>;

function SingleChoiceField({ input, value, onChange }: InputFieldProps) {
  return (
    <RadioGroup value={String(value || '')} onValueChange={onChange} className="space-y-2">
      {input.options?.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <RadioGroupItem value={option.value} id={`${input.id}-${option.value}`} />
          <Label htmlFor={`${input.id}-${option.value}`} className="font-normal cursor-pointer">
            {option.label}
          </Label>
        </div>
      ))}
    </RadioGroup>
  );
}

function MultiChoiceField({ input, value, onChange }: InputFieldProps) {
  const selected = Array.isArray(value) ? value : [];
  return (
    <div className="space-y-2">
      {input.options?.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <Checkbox
            id={`${input.id}-${option.value}`}
            checked={selected.includes(option.value)}
            onCheckedChange={(checked) => {
              onChange(checked ? [...selected, option.value] : selected.filter((v) => v !== option.value));
            }}
          />
          <Label htmlFor={`${input.id}-${option.value}`} className="font-normal cursor-pointer">
            {option.label}
          </Label>
        </div>
      ))}
    </div>
  );
}

interface InputFieldProps {
  input: HitlInterruptData['inputs'][0];
  value: ResponseValue;
  onChange: (value: ResponseValue) => void;
}

function InputField({ input, value, onChange }: InputFieldProps) {
  switch (input.input_type) {
    case 'single_choice':
      return <SingleChoiceField input={input} value={value} onChange={onChange} />;
    case 'multi_choice':
      return <MultiChoiceField input={input} value={value} onChange={onChange} />;
    case 'text':
      return <Textarea value={String(value || '')} onChange={(e) => onChange(e.target.value)} placeholder="Enter your response..." rows={3} />;
    case 'number':
      return <Input type="number" value={value === '' || value === undefined ? '' : Number(value)} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : '')} placeholder="Enter a number..." />;
    case 'boolean':
      return (
        <div className="flex items-center space-x-2">
          <Checkbox id={input.id} checked={Boolean(value)} onCheckedChange={(checked) => onChange(checked === true)} />
          <Label htmlFor={input.id} className="font-normal cursor-pointer">Yes</Label>
        </div>
      );
    case 'table':
      return <TableInput input={input} value={Array.isArray(value) ? value as Array<Record<string, unknown>> : []} onChange={(v) => onChange(v as unknown as ResponseValue)} />;
    default:
      return <Input value={String(value || '')} onChange={(e) => onChange(e.target.value)} placeholder="Enter your response..." />;
  }
}

/**
 * Renders a single display item with adaptive layout
 * - Short values: side-by-side layout (label | value)
 * - Long/markdown values: stacked layout with rich rendering
 */
function DisplayItem({ item }: { item: { label: string; value: unknown } }) {
  const analysis = analyzeContent(item.value);

  // For short, non-markdown values: compact side-by-side layout
  if (!analysis.isLong && !analysis.isMarkdown) {
    return (
      <div className="flex flex-col sm:flex-row sm:justify-between gap-2 sm:gap-4">
        <span className="text-sm text-muted-foreground shrink-0">{item.label}:</span>
        <DisplayValue value={item.value} className="max-w-full sm:max-w-[60%]" />
      </div>
    );
  }

  // For long or markdown values: stacked layout with rich rendering
  return (
    <div className="space-y-2">
      <span className="text-sm font-medium text-muted-foreground">{item.label}</span>
      <RichDisplayValue value={item.value} analysis={analysis} />
    </div>
  );
}

/**
 * Displays context data from the workflow
 */
function DisplaySection({ display }: { display: HitlInterruptData['display'] }) {
  if (!display || display.length === 0) {
    return null;
  }

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-muted-foreground">Context</h4>
      <div className="bg-muted/50 rounded-lg p-4 space-y-4">
        {display.map((item, index) => (
          <DisplayItem key={index} item={item} />
        ))}
      </div>
    </div>
  );
}

function buildInitialResponses(inputs: HitlInterruptData['inputs']): Record<string, ResponseValue> {
  const initial: Record<string, ResponseValue> = {};
  for (const input of inputs) {
    if (input.input_type === 'multi_choice') {
      initial[input.id] = [];
    } else if (input.input_type === 'table') {
      initial[input.id] = (input.rows || []).map(() =>
        Object.fromEntries((input.columns || []).map((col) => [col.id, col.input_type === 'boolean' ? false : '']))
      );
    } else if (input.input_type === 'boolean') {
      initial[input.id] = false;
    } else {
      initial[input.id] = '';
    }
  }
  return initial;
}

export function HITLResponsePanel({
  interruptData,
  onSubmit,
  isSubmitting = false,
  error,
}: HITLResponsePanelProps) {
  const [responses, setResponses] = useState<Record<string, ResponseValue>>(() => buildInitialResponses(interruptData.inputs));

  const updateResponse = useCallback((id: string, value: ResponseValue) => {
    setResponses((prev) => ({ ...prev, [id]: value }));
  }, []);

  const handleSubmit = useCallback(() => {
    // Build the payload
    const payload: HitlResumePayload = {
      responses: responses as Record<string, string | number | boolean | string[]>,
    };
    onSubmit(payload);
  }, [responses, onSubmit]);

  // Check if all required fields are filled
  const isValid = interruptData.inputs.every((input) => {
    if (!input.required) return true;
    const value = responses[input.id];
    if (input.input_type === 'table') {
      // Table is valid if it has rows (they come from data, so always valid if rows exist)
      return Array.isArray(value) && value.length > 0;
    }
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'boolean') return true; // Boolean is always "filled"
    return value !== '' && value !== undefined;
  });

  return (
    <Card className="border-amber-500/50 bg-amber-500/5">
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-amber-500/20 flex items-center justify-center">
            <UserCheck className="w-4 h-4 text-amber-500" />
          </div>
          <div>
            <CardTitle className="text-lg">{interruptData.title}</CardTitle>
            {interruptData.description && (
              <CardDescription>{interruptData.description}</CardDescription>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Display context data */}
        <DisplaySection display={interruptData.display} />

        {/* Input fields */}
        <div className="space-y-4">
          {interruptData.inputs.map((input) => (
            <div key={input.id} className="space-y-2">
              <Label className="text-sm font-medium">
                {input.question}
                {input.required && <span className="text-destructive ml-1">*</span>}
              </Label>
              <InputField
                input={input}
                value={responses[input.id]}
                onChange={(value) => updateResponse(input.id, value)}
              />
            </div>
          ))}
        </div>

        {/* Error message */}
        {error && (
          <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="w-4 h-4" />
            <span>{error.message || 'Failed to submit response'}</span>
          </div>
        )}

        {/* Submit button */}
        <div className="flex justify-end pt-2">
          <Button
            onClick={handleSubmit}
            disabled={!isValid || isSubmitting}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Submitting...
              </>
            ) : (
              'Submit Response'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
