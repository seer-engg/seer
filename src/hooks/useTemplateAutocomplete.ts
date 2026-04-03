import { useCallback, useMemo, useState } from 'react';
import type { KeyboardEvent, RefObject } from 'react';
import {
  type VariableTreeItem,
  getVariableTreeItems,
  filterTreeItems,
} from '@/components/workflows/block-config/helpers/variableTreeUtils';

export interface AutocompleteContext {
  inputId: string;
  ref: RefObject<HTMLInputElement | HTMLTextAreaElement>;
  value: string;
  onChange: (value: string) => void;
}

interface HandleKeyDownOptions {
  allowShiftEnter?: boolean;
}

export const useTemplateAutocomplete = (availableVariables: string[]) => {
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [partialVariable, setPartialVariable] = useState('');
  const [autocompleteContext, setAutocompleteContext] = useState<AutocompleteContext | null>(null);
  const [currentPath, setCurrentPath] = useState('');

  const currentLevelItems = useMemo(() => {
    const items = getVariableTreeItems(availableVariables, currentPath);
    return filterTreeItems(items, partialVariable);
  }, [availableVariables, currentPath, partialVariable]);

  const closeAutocomplete = useCallback(() => {
    setShowAutocomplete(false);
    setPartialVariable('');
    setAutocompleteContext(null);
    setCurrentPath('');
  }, []);

  const drillInto = useCallback((item: VariableTreeItem) => {
    setCurrentPath(item.fullPath);
    setSelectedIndex(0);
    setPartialVariable('');
  }, []);

  const navigateUp = useCallback(() => {
    setCurrentPath((prev) => {
      if (!prev) return prev;
      const lastDot = prev.lastIndexOf('.');
      return lastDot === -1 ? '' : prev.slice(0, lastDot);
    });
    setSelectedIndex(0);
  }, []);

  const navigateTo = useCallback((path: string) => {
    setCurrentPath(path);
    setSelectedIndex(0);
  }, []);

  const checkForAutocomplete = useCallback(
    (value: string, cursorPosition: number, context?: AutocompleteContext) => {
      const textBeforeCursor = value.substring(0, cursorPosition);
      const lastSlash = textBeforeCursor.lastIndexOf('/');

      if (lastSlash !== -1) {
        const textAfterOpen = textBeforeCursor.substring(lastSlash + 1);
        const hasClosing = textAfterOpen.includes('}');

        if (!hasClosing) {
          const partial = textAfterOpen.trim();
          setPartialVariable(partial);
          setShowAutocomplete(true);
          setSelectedIndex(0);
          setCurrentPath('');
          if (context) {
            setAutocompleteContext({ ...context, value });
          }
          return;
        }
      }

      setShowAutocomplete(false);
      setPartialVariable('');
      if (!context) {
        setAutocompleteContext(null);
      }
    },
    []
  );

  const insertVariable = useCallback(
    (variable: string, fallbackContext?: AutocompleteContext) => {
      const context = autocompleteContext || fallbackContext;
      if (!context) return;

      const { ref, value, onChange } = context;
      const input = ref.current;
      if (!input) return;

      const cursorPos = input.selectionStart || 0;
      const textBeforeCursor = value.substring(0, cursorPos);
      const textAfterCursor = value.substring(cursorPos);

      const lastSlash = textBeforeCursor.lastIndexOf('/');
      if (lastSlash === -1) return;

      const beforeSlash = value.substring(0, lastSlash);
      const newValue = beforeSlash + '${' + variable + '}' + textAfterCursor;

      onChange(newValue);
      setShowAutocomplete(false);
      setPartialVariable('');
      setAutocompleteContext(null);
      setCurrentPath('');

      setTimeout(() => {
        const newCursorPos = lastSlash + variable.length + 3;
        input.setSelectionRange(newCursorPos, newCursorPos);
        input.focus();
      }, 0);
    },
    [autocompleteContext]
  );

  const handleNavigationKey = useCallback(
    (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      const selectedItem = currentLevelItems[selectedIndex];

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, currentLevelItems.length - 1));
          return true;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          return true;
        case 'ArrowRight':
          if (selectedItem?.hasChildren) {
            e.preventDefault();
            drillInto(selectedItem);
          }
          return true;
        case 'ArrowLeft':
          if (currentPath) {
            e.preventDefault();
            navigateUp();
          }
          return true;
        case 'Escape':
          e.preventDefault();
          if (currentPath) { navigateUp(); } else { closeAutocomplete(); }
          return true;
        default:
          return false;
      }
    },
    [closeAutocomplete, currentLevelItems, currentPath, drillInto, navigateUp, selectedIndex]
  );

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>, options: HandleKeyDownOptions = {}) => {
      if (!showAutocomplete) return;

      if (handleNavigationKey(e)) return;

      if (e.key === 'Enter') {
        if (!options.allowShiftEnter && e.shiftKey) return;
        const item = currentLevelItems[selectedIndex];
        if (item) {
          e.preventDefault();
          insertVariable(item.fullPath);
        }
      }
    },
    [currentLevelItems, handleNavigationKey, insertVariable, selectedIndex, showAutocomplete]
  );

  return {
    autocompleteContext,
    checkForAutocomplete,
    closeAutocomplete,
    currentLevelItems,
    currentPath,
    drillInto,
    handleKeyDown,
    insertVariable,
    navigateTo,
    navigateUp,
    selectedIndex,
    setAutocompleteContext,
    setSelectedIndex,
    showAutocomplete,
  };
};
