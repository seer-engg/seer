import { useRef } from 'react';
import { Input } from '@/components/ui/input';
import { VariableAutocompleteDropdown } from './VariableAutocompleteDropdown';
import type { TemplateAutocompleteControls } from '../types';
import type { AutocompleteContext } from '../../../../hooks/useTemplateAutocomplete';

export interface AutocompleteInputProps {
  id: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  templateAutocomplete: TemplateAutocompleteControls;
  className?: string;
  disabled?: boolean;
  type?: string;
  step?: string | number;
  min?: number;
  max?: number;
}

/**
 * Input component with template variable autocomplete support.
 * Provides ${variable} suggestions while typing.
 */
export function AutocompleteInput({
  id,
  value,
  onChange,
  placeholder,
  templateAutocomplete,
  className = '',
  disabled,
  type,
  step,
  min,
  max,
}: AutocompleteInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

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

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    const cursorPosition = e.target.selectionStart ?? e.target.value.length;
    onChange(newValue);

    const context: AutocompleteContext = {
      inputId: id,
      ref: inputRef,
      value: newValue,
      onChange,
    };
    checkForAutocomplete(newValue, cursorPosition, context);
  };

  const handleFocus = () => {
    setAutocompleteContext({
      inputId: id,
      ref: inputRef,
      value,
      onChange,
    });
  };

  const handleBlur = () => {
    // Delay to allow click on dropdown item
    setTimeout(() => closeAutocomplete(), 200);
  };

  return (
    <div className="relative">
      <Input
        ref={inputRef}
        id={id}
        type={type}
        step={step}
        min={min}
        max={max}
        value={value}
        onChange={handleChange}
        onFocus={handleFocus}
        onKeyDown={handleKeyDown}
        onBlur={handleBlur}
        placeholder={placeholder}
        className={className}
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
