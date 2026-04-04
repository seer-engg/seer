import { useRef, type RefObject } from 'react';
import { Textarea } from '@/components/ui/textarea';
import { VariableAutocompleteDropdown } from './VariableAutocompleteDropdown';
import type { TemplateAutocompleteControls } from '../types';
import type { AutocompleteContext } from '../../../../hooks/useTemplateAutocomplete';

export interface AutocompleteTextareaProps {
  id: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  templateAutocomplete: TemplateAutocompleteControls;
  className?: string;
  style?: React.CSSProperties;
  rows?: number;
  externalRef?: RefObject<HTMLTextAreaElement>;
  disabled?: boolean;
}

/**
 * Textarea component with template variable autocomplete support.
 * Provides ${variable} suggestions while typing.
 */
export function AutocompleteTextarea({
  id,
  value,
  onChange,
  placeholder,
  templateAutocomplete,
  className = '',
  style,
  rows = 4,
  externalRef,
  disabled,
}: AutocompleteTextareaProps) {
  const internalRef = useRef<HTMLTextAreaElement>(null);
  const textareaRef = externalRef || internalRef;

  const {
    autocompleteContext,
    checkForAutocomplete,
    closeAutocomplete,
    currentLevelItems,
    currentPath,
    drillInto,
    handleKeyDown,
    insertVariable,
    navigateTo,
    selectedIndex,
    setAutocompleteContext,
    showAutocomplete,
  } = templateAutocomplete;

  const dropdownVisible = showAutocomplete && autocompleteContext?.inputId === id;

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    const cursorPosition = e.target.selectionStart ?? e.target.value.length;
    onChange(newValue);

    const context: AutocompleteContext = {
      inputId: id,
      ref: textareaRef,
      value: newValue,
      onChange,
    };
    checkForAutocomplete(newValue, cursorPosition, context);
  };

  const handleFocus = () => {
    const cursorPosition = textareaRef.current?.selectionStart ?? 0;
    const context: AutocompleteContext = {
      inputId: id,
      ref: textareaRef,
      value,
      onChange,
    };
    checkForAutocomplete(value, cursorPosition, context);
  };

  const handleBlur = () => {
    // Delay to allow click on dropdown item
    setTimeout(() => closeAutocomplete(), 200);
  };

  return (
    <div className="relative">
      <Textarea
        ref={textareaRef}
        id={id}
        value={value}
        onChange={handleChange}
        onFocus={handleFocus}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        placeholder={placeholder}
        className={className}
        style={style}
        rows={rows}
        disabled={disabled}
      />
      <VariableAutocompleteDropdown
        visible={dropdownVisible}
        items={currentLevelItems}
        selectedIndex={selectedIndex}
        currentPath={currentPath}
        onSelect={insertVariable}
        onDrillInto={drillInto}
        onNavigateTo={navigateTo}
      />
    </div>
  );
}
