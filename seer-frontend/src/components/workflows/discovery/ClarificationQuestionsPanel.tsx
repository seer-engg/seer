import { useState, useCallback } from 'react';
import { ChevronLeft, ChevronRight, Send } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';
import { Progress } from '@/components/ui/progress';
import type { ClarificationQuestion, ClarificationAnswer, ClarificationAnswers } from '@/types/discovery';
import { cn } from '@/lib/utils';
import { ResourcePickerQuestion } from './ResourcePickerQuestion';
import { AccountPickerQuestion } from './AccountPickerQuestion';

interface QuestionState {
  selectedValues: string[];
  customInput: string;
  /** Display names for resource picker selections (value -> display name) */
  displayNames?: Record<string, string>;
}

// --- Utility Functions ---

function getHasWildcard(selectedValues: string[], options: ClarificationQuestion['options']): boolean {
  return selectedValues.some((value) => options.find((opt) => opt.value === value)?.is_wildcard);
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

function getHelperText(isSingleChoice: boolean, minSelections: number, maxSelections: number | null): string {
  if (isSingleChoice) return '';
  if (minSelections === maxSelections) return `Select exactly ${minSelections} option${minSelections > 1 ? 's' : ''}`;
  if (maxSelections) return `Select ${minSelections}-${maxSelections} options`;
  return `Select at least ${minSelections} option${minSelections > 1 ? 's' : ''}`;
}

function buildAnswersArray(
  questions: ClarificationQuestion[],
  answers: Map<string, QuestionState>
): ClarificationAnswer[] {
  return questions.map((q) => {
    const state = answers.get(q.question_id) || { selectedValues: [], customInput: '' };
    const qHasWildcard = getHasWildcard(state.selectedValues, q.options);
    return {
      question_id: q.question_id,
      selected_values: state.selectedValues,
      custom_input: qHasWildcard ? state.customInput.trim() || null : null,
    };
  });
}

/**
 * Builds the dependsOnValues map for a resource picker question.
 *
 * The backend sends `depends_on` as a question_id (e.g., "q_67b3198c") and
 * `depends_on_field` as the API parameter name (e.g., "guild_id").
 *
 * The ResourcePicker component expects dependsOnValues as:
 * { [depends_on_field]: selected_value }
 *
 * Example:
 * - Question depends_on: "q_67b3198c" (guild question)
 * - Question depends_on_field: "guild_id"
 * - User selected guild: "guild_123"
 * - Returns: { guild_id: "guild_123" }
 */
function buildDependsOnValues(
  question: ClarificationQuestion,
  answers: Map<string, QuestionState>
): Record<string, string> | undefined {
  if (!question.depends_on || !question.depends_on_field) {
    return undefined;
  }

  const dependencyAnswer = answers.get(question.depends_on);
  if (!dependencyAnswer || dependencyAnswer.selectedValues.length === 0) {
    return undefined;
  }

  // Use the first selected value (resource pickers are typically single-select)
  return {
    [question.depends_on_field]: dependencyAnswer.selectedValues[0],
  };
}

// --- Sub-Components ---

function SingleChoiceOptions({
  options,
  selectedValue,
  onValueChange,
  isLoading,
}: {
  options: ClarificationQuestion['options'];
  selectedValue: string;
  onValueChange: (value: string) => void;
  isLoading: boolean;
}) {
  return (
    <RadioGroup value={selectedValue} onValueChange={onValueChange} disabled={isLoading} className="space-y-2">
      {options.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <RadioGroupItem value={option.value} id={`radio-${option.value}`} />
          <Label htmlFor={`radio-${option.value}`} className={cn('text-sm cursor-pointer', isLoading && 'opacity-50 cursor-not-allowed')}>
            {option.label}
            {option.is_wildcard && <span className="text-muted-foreground ml-1">(specify)</span>}
          </Label>
        </div>
      ))}
    </RadioGroup>
  );
}

function MultiChoiceOptions({
  options,
  selectedValues,
  onToggle,
  isLoading,
  maxSelections,
  onValidationError,
}: {
  options: ClarificationQuestion['options'];
  selectedValues: string[];
  onToggle: (value: string, checked: boolean) => void;
  isLoading: boolean;
  maxSelections: number | null;
  onValidationError: (error: string | null) => void;
}) {
  const handleToggle = (value: string, checked: boolean) => {
    if (checked && maxSelections && selectedValues.length >= maxSelections) {
      onValidationError(`You can select at most ${maxSelections} option${maxSelections > 1 ? 's' : ''}`);
      return;
    }
    onValidationError(null);
    onToggle(value, checked);
  };

  return (
    <div className="space-y-2">
      {options.map((option) => (
        <div key={option.value} className="flex items-center space-x-2">
          <Checkbox
            id={`checkbox-${option.value}`}
            checked={selectedValues.includes(option.value)}
            onCheckedChange={(checked) => handleToggle(option.value, checked as boolean)}
            disabled={isLoading}
          />
          <Label htmlFor={`checkbox-${option.value}`} className={cn('text-sm cursor-pointer', isLoading && 'opacity-50 cursor-not-allowed')}>
            {option.label}
            {option.is_wildcard && <span className="text-muted-foreground ml-1">(specify)</span>}
          </Label>
        </div>
      ))}
    </div>
  );
}

function WizardProgress({ current, total }: { current: number; total: number }) {
  const progressPercent = ((current + 1) / total) * 100;
  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-muted-foreground">Question {current + 1} of {total}</span>
        <span className="text-xs text-muted-foreground">{Math.round(progressPercent)}%</span>
      </div>
      <Progress value={progressPercent} className="h-1" />
    </div>
  );
}

function WizardNavigation({
  isFirst,
  isLast,
  canProceed,
  isLoading,
  onBack,
  onNext,
  onSubmit,
}: {
  isFirst: boolean;
  isLast: boolean;
  canProceed: boolean;
  isLoading: boolean;
  onBack: () => void;
  onNext: () => void;
  onSubmit: () => void;
}) {
  return (
    <div className="flex gap-2 mt-4">
      {!isFirst && (
        <Button variant="outline" size="sm" onClick={onBack} disabled={isLoading} className="flex-1">
          <ChevronLeft className="w-4 h-4 mr-1" />
          Back
        </Button>
      )}
      {isLast ? (
        <Button size="sm" onClick={onSubmit} disabled={isLoading || !canProceed} className={cn('flex-1', isFirst && 'w-full')}>
          {isLoading ? 'Submitting...' : 'Submit'}
          {!isLoading && <Send className="w-4 h-4 ml-1" />}
        </Button>
      ) : (
        <Button size="sm" onClick={onNext} disabled={isLoading || !canProceed} className={cn('flex-1', isFirst && 'w-full')}>
          Next
          <ChevronRight className="w-4 h-4 ml-1" />
        </Button>
      )}
    </div>
  );
}

function QuestionContent({
  question,
  currentState,
  isLoading,
  maxSelections,
  dependsOnValues,
  onSingleChange,
  onMultiToggle,
  onResourceChange,
  onAccountChange,
  onValidationError,
}: {
  question: ClarificationQuestion;
  currentState: QuestionState;
  isLoading: boolean;
  maxSelections: number | null;
  dependsOnValues?: Record<string, string>;
  onSingleChange: (value: string) => void;
  onMultiToggle: (value: string, checked: boolean) => void;
  onResourceChange: (value: string, displayName?: string) => void;
  onAccountChange: (connectionId: string, displayName?: string) => void;
  onValidationError: (error: string | null) => void;
}) {
  if (question.question_type === 'resource_picker') {
    return (
      <ResourcePickerQuestion
        question={question}
        selectedValue={currentState.selectedValues[0] || ''}
        onValueChange={onResourceChange}
        isLoading={isLoading}
        dependsOnValues={dependsOnValues}
      />
    );
  }

  if (question.question_type === 'account_picker') {
    return (
      <AccountPickerQuestion
        question={question}
        value={currentState.selectedValues[0]}
        onChange={onAccountChange}
        isLoading={isLoading}
      />
    );
  }

  if (question.question_type === 'single_choice') {
    return (
      <SingleChoiceOptions
        options={question.options}
        selectedValue={currentState.selectedValues[0] || ''}
        onValueChange={onSingleChange}
        isLoading={isLoading}
      />
    );
  }

  return (
    <MultiChoiceOptions
      options={question.options}
      selectedValues={currentState.selectedValues}
      onToggle={onMultiToggle}
      isLoading={isLoading}
      maxSelections={maxSelections}
      onValidationError={onValidationError}
    />
  );
}

// --- Custom Hook ---

function useWizardState(questions: ClarificationQuestion[]) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<Map<string, QuestionState>>(() => new Map());
  const [validationError, setValidationError] = useState<string | null>(null);

  const currentQuestion = questions[currentIndex];
  const currentState = answers.get(currentQuestion.question_id) || { selectedValues: [], customInput: '' };

  const updateCurrentAnswer = useCallback((updates: Partial<QuestionState>) => {
    setAnswers((prev) => {
      const next = new Map(prev);
      const current = next.get(currentQuestion.question_id) || { selectedValues: [], customInput: '' };
      next.set(currentQuestion.question_id, { ...current, ...updates });
      return next;
    });
    setValidationError(null);
  }, [currentQuestion.question_id]);

  const goNext = useCallback(() => setCurrentIndex((prev) => Math.min(prev + 1, questions.length - 1)), [questions.length]);
  const goBack = useCallback(() => { setValidationError(null); setCurrentIndex((prev) => Math.max(prev - 1, 0)); }, []);

  return {
    currentIndex, currentQuestion, currentState, answers,
    validationError, setValidationError,
    updateCurrentAnswer, goNext, goBack,
    isFirst: currentIndex === 0,
    isLast: currentIndex === questions.length - 1,
  };
}

// --- Main Component ---

/** Computes validation state for the current question */
function getQuestionValidation(
  question: ClarificationQuestion,
  state: QuestionState,
  answers: Map<string, QuestionState>
) {
  const isResourcePicker = question.question_type === 'resource_picker';
  const isAccountPicker = question.question_type === 'account_picker';
  const minSelections = question.min_selections || 1;
  const maxSelections = question.max_selections || null;
  const hasWildcard = !isResourcePicker && !isAccountPicker && getHasWildcard(state.selectedValues, question.options || []);
  const dependsOnValues = isResourcePicker ? buildDependsOnValues(question, answers) : undefined;
  const hasMissingDependency = isResourcePicker && question.depends_on && !dependsOnValues;

  let canProceed: boolean;
  if (isResourcePicker) {
    canProceed = state.selectedValues.length > 0 && !hasMissingDependency;
  } else if (isAccountPicker) {
    canProceed = state.selectedValues.length > 0;
  } else {
    canProceed = state.selectedValues.length >= minSelections && (!hasWildcard || state.customInput.trim().length > 0);
  }

  return { isResourcePicker, isAccountPicker, minSelections, maxSelections, hasWildcard, dependsOnValues, canProceed };
}

export function ClarificationQuestionsPanel({
  questions,
  onSubmit,
  isLoading,
}: {
  questions: ClarificationQuestion[];
  onSubmit: (answers: ClarificationAnswers) => void;
  isLoading: boolean;
}) {
  const wizard = useWizardState(questions);
  const { currentQuestion, currentState, validationError, setValidationError } = wizard;

  const { isResourcePicker, isAccountPicker, minSelections, maxSelections, hasWildcard, dependsOnValues, canProceed } =
    getQuestionValidation(currentQuestion, currentState, wizard.answers);

  const isSingleChoice = currentQuestion.question_type === 'single_choice';

  const handleValidateAndProceed = (action: () => void) => {
    if (isResourcePicker) {
      if (currentState.selectedValues.length === 0) { setValidationError('Please select a resource'); return; }
    } else if (isAccountPicker) {
      if (currentState.selectedValues.length === 0) { setValidationError('Please select an account'); return; }
    } else {
      const error = validateAnswer(currentState.selectedValues, currentState.customInput, minSelections, maxSelections, hasWildcard);
      if (error) { setValidationError(error); return; }
    }
    action();
  };

  const handleSubmit = () => handleValidateAndProceed(() => onSubmit({ answers: buildAnswersArray(questions, wizard.answers) }));
  const handleNext = () => handleValidateAndProceed(wizard.goNext);

  return (
    <div className="border border-border rounded-lg p-4 bg-muted/30 mb-4">
      <WizardProgress current={wizard.currentIndex} total={questions.length} />
      <p className="text-sm font-medium mb-3">{currentQuestion.question}</p>

      <QuestionContent
        question={currentQuestion}
        currentState={currentState}
        isLoading={isLoading}
        maxSelections={maxSelections}
        dependsOnValues={dependsOnValues}
        onSingleChange={(v) => wizard.updateCurrentAnswer({ selectedValues: [v] })}
        onMultiToggle={(v, c) => wizard.updateCurrentAnswer({ selectedValues: c ? [...currentState.selectedValues, v] : currentState.selectedValues.filter((x) => x !== v) })}
        onResourceChange={(v, d) => wizard.updateCurrentAnswer({ selectedValues: [v], displayNames: d ? { [v]: d } : undefined })}
        onAccountChange={(v, d) => wizard.updateCurrentAnswer({ selectedValues: [v], displayNames: d ? { [v]: d } : undefined })}
        onValidationError={setValidationError}
      />

      {hasWildcard && (
        <div className="mt-4 space-y-2">
          <Label htmlFor="custom-input" className="text-xs text-muted-foreground">Please specify:</Label>
          <Input
            id="custom-input"
            value={currentState.customInput}
            onChange={(e) => wizard.updateCurrentAnswer({ customInput: e.target.value })}
            placeholder="Enter your answer..."
            disabled={isLoading}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey && canProceed) { e.preventDefault(); if (wizard.isLast) { handleSubmit(); } else { handleNext(); } }}}
            className="text-sm"
          />
        </div>
      )}

      {validationError && <p className="text-xs text-destructive mt-2">{validationError}</p>}
      {!isSingleChoice && !isResourcePicker && !isAccountPicker && <p className="text-xs text-muted-foreground mt-2">{getHelperText(isSingleChoice, minSelections, maxSelections)}</p>}

      <WizardNavigation isFirst={wizard.isFirst} isLast={wizard.isLast} canProceed={canProceed} isLoading={isLoading} onBack={wizard.goBack} onNext={handleNext} onSubmit={handleSubmit} />
    </div>
  );
}
