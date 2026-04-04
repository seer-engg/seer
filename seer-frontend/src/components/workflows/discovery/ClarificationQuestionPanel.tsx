import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import type { ClarificationQuestion, ClarificationAnswer } from '@/types/discovery';
import { cn } from '@/lib/utils';

interface ClarificationQuestionPanelProps {
  question: ClarificationQuestion;
  onAnswer: (answer: ClarificationAnswer) => void;
  isLoading: boolean;
}

function getHasWildcard(selectedValues: string[], options: ClarificationQuestion['options']): boolean {
  return selectedValues.some((value) =>
    options.find((opt) => opt.value === value)?.is_wildcard
  );
}

function validateAnswer(
  selectedValues: string[],
  customInput: string,
  minSelections: number,
  maxSelections: number | null,
  hasWildcard: boolean
): string | null {
  if (selectedValues.length < minSelections) {
    return `Please select at least ${minSelections} option${minSelections > 1 ? 's' : ''}`;
  }

  if (maxSelections && selectedValues.length > maxSelections) {
    return `Please select at most ${maxSelections} option${maxSelections > 1 ? 's' : ''}`;
  }

  if (hasWildcard && !customInput.trim()) {
    return 'Please provide custom input for the selected option';
  }

  return null;
}

function getHelperText(
  isSingleChoice: boolean,
  minSelections: number,
  maxSelections: number | null
): string {
  if (isSingleChoice) return '';

  if (minSelections === maxSelections) {
    return `Select exactly ${minSelections} option${minSelections > 1 ? 's' : ''}`;
  }

  if (maxSelections) {
    return `Select ${minSelections}-${maxSelections} options`;
  }

  return `Select at least ${minSelections} option${minSelections > 1 ? 's' : ''}`;
}

interface SingleChoiceOptionsProps {
  options: ClarificationQuestion['options'];
  selectedValue: string;
  onValueChange: (value: string) => void;
  isLoading: boolean;
}

function SingleChoiceOptions({ options, selectedValue, onValueChange, isLoading }: SingleChoiceOptionsProps) {
  return (
    <RadioGroup value={selectedValue} onValueChange={onValueChange} disabled={isLoading} className="space-y-2">
      {options.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <RadioGroupItem value={option.value} id={`radio-${option.value}`} />
          <Label
            htmlFor={`radio-${option.value}`}
            className={cn('text-sm cursor-pointer', isLoading && 'opacity-50 cursor-not-allowed')}
          >
            {option.label}
            {option.is_wildcard && <span className="text-muted-foreground ml-1">(specify)</span>}
          </Label>
        </div>
      ))}
    </RadioGroup>
  );
}

interface MultiChoiceOptionsProps {
  options: ClarificationQuestion['options'];
  selectedValues: string[];
  onToggle: (value: string, checked: boolean) => void;
  isLoading: boolean;
}

function MultiChoiceOptions({ options, selectedValues, onToggle, isLoading }: MultiChoiceOptionsProps) {
  return (
    <div className="space-y-2">
      {options.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <Checkbox
            id={`checkbox-${option.value}`}
            checked={selectedValues.includes(option.value)}
            onCheckedChange={(checked) => onToggle(option.value, checked as boolean)}
            disabled={isLoading}
          />
          <Label
            htmlFor={`checkbox-${option.value}`}
            className={cn('text-sm cursor-pointer', isLoading && 'opacity-50 cursor-not-allowed')}
          >
            {option.label}
            {option.is_wildcard && <span className="text-muted-foreground ml-1">(specify)</span>}
          </Label>
        </div>
      ))}
    </div>
  );
}

function useQuestionState(question: ClarificationQuestion) {
  const [selectedValues, setSelectedValues] = useState<string[]>([]);
  const [customInput, setCustomInput] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedValues([]);
    setCustomInput('');
    setShowCustomInput(false);
    setValidationError(null);
  }, [question.question_id]);

  useEffect(() => {
    const hasWildcard = getHasWildcard(selectedValues, question.options);
    setShowCustomInput(hasWildcard);
    if (!hasWildcard) setCustomInput('');
  }, [selectedValues, question.options]);

  return { selectedValues, setSelectedValues, customInput, setCustomInput, showCustomInput, validationError, setValidationError };
}

export function ClarificationQuestionPanel({
  question,
  onAnswer,
  isLoading,
}: ClarificationQuestionPanelProps) {
  const { selectedValues, setSelectedValues, customInput, setCustomInput, showCustomInput, validationError, setValidationError } =
    useQuestionState(question);

  const isSingleChoice = question.question_type === 'single_choice';
  const minSelections = question.min_selections || 1;
  const maxSelections = question.max_selections || null;

  const handleSingleChoiceChange = (value: string) => {
    setSelectedValues([value]);
    setValidationError(null);
  };

  const handleMultiChoiceToggle = (value: string, checked: boolean) => {
    setValidationError(null);
    if (checked) {
      if (maxSelections && selectedValues.length >= maxSelections) {
        setValidationError(`You can select at most ${maxSelections} option${maxSelections > 1 ? 's' : ''}`);
        return;
      }
      setSelectedValues([...selectedValues, value]);
    } else {
      setSelectedValues(selectedValues.filter((v) => v !== value));
    }
  };

  const handleSubmit = () => {
    setValidationError(null);
    const hasWildcard = getHasWildcard(selectedValues, question.options);
    const error = validateAnswer(selectedValues, customInput, minSelections, maxSelections, hasWildcard);
    if (error) {
      setValidationError(error);
      return;
    }
    onAnswer({
      question_id: question.question_id,
      selected_values: selectedValues,
      custom_input: hasWildcard ? customInput.trim() : null,
    });
  };

  const canSubmit = selectedValues.length >= minSelections && (!showCustomInput || customInput.trim().length > 0);

  return (
    <div className="border border-border rounded-lg p-4 bg-muted/30 mb-4">
      <p className="text-sm font-medium mb-3">{question.question}</p>

      {isSingleChoice ? (
        <SingleChoiceOptions
          options={question.options}
          selectedValue={selectedValues[0] || ''}
          onValueChange={handleSingleChoiceChange}
          isLoading={isLoading}
        />
      ) : (
        <MultiChoiceOptions
          options={question.options}
          selectedValues={selectedValues}
          onToggle={handleMultiChoiceToggle}
          isLoading={isLoading}
        />
      )}

      {showCustomInput && (
        <div className="mt-4 space-y-2">
          <Label htmlFor="custom-input" className="text-xs text-muted-foreground">Please specify:</Label>
          <Input
            id="custom-input"
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            placeholder="Enter your answer..."
            disabled={isLoading}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && canSubmit) {
                e.preventDefault();
                handleSubmit();
              }
            }}
            className="text-sm"
          />
        </div>
      )}

      {validationError && <p className="text-xs text-destructive mt-2">{validationError}</p>}

      <Button onClick={handleSubmit} disabled={isLoading || !canSubmit} className="mt-4 w-full" size="sm">
        {isLoading ? 'Submitting...' : 'Submit'}
      </Button>

      {!isSingleChoice && (
        <p className="text-xs text-muted-foreground mt-2 text-center">
          {getHelperText(isSingleChoice, minSelections, maxSelections)}
        </p>
      )}
    </div>
  );
}
