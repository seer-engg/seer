import React, { createContext, useContext, useEffect, useState, useCallback, useMemo, useRef } from 'react';

export interface ShortcutDefinition {
  key: string;
  modifiers?: {
    ctrl?: boolean;
    meta?: boolean;
    shift?: boolean;
    alt?: boolean;
  };
  handler: () => void;
  category: string;
  description: string;
  scope?: 'global' | 'dialog';
  enabled?: boolean;
}

/**
 * Internal shortcut type that stores a stable handler wrapper.
 * The handlerRef allows us to always call the latest handler
 * without re-registering the shortcut.
 */
interface ShortcutWithId extends Omit<ShortcutDefinition, 'handler'> {
  id: string;
  handlerRef: React.MutableRefObject<() => void>;
}

/**
 * Stable shortcut definition used for registration.
 * Uses a ref for the handler to avoid re-registration when handler changes.
 */
interface StableShortcutDefinition extends Omit<ShortcutDefinition, 'handler'> {
  handlerRef: React.MutableRefObject<() => void>;
}

interface KeyboardShortcutContextValue {
  shortcuts: ShortcutWithId[];
  registerShortcut: (shortcut: StableShortcutDefinition) => string;
  unregisterShortcut: (id: string) => void;
}

const KeyboardShortcutContext = createContext<KeyboardShortcutContextValue | null>(null);

let shortcutIdCounter = 0;
const generateId = () => `shortcut-${++shortcutIdCounter}`;

export function KeyboardShortcutProvider({ children }: { children: React.ReactNode }) {
  const [shortcuts, setShortcuts] = useState<Map<string, ShortcutWithId>>(new Map());

  const registerShortcut = useCallback((shortcut: StableShortcutDefinition) => {
    const id = generateId();
    const shortcutWithId: ShortcutWithId = { ...shortcut, id };
    setShortcuts((prev) => {
      const next = new Map(prev);
      next.set(id, shortcutWithId);
      return next;
    });
    return id;
  }, []);

  const unregisterShortcut = useCallback((id: string) => {
    setShortcuts((prev) => {
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const getModifierState = (event: KeyboardEvent) => ({
    ctrl: event.ctrlKey || event.metaKey,
    shift: event.shiftKey,
    alt: event.altKey,
  });

  const getRequiredModifiers = (modifiers?: ShortcutDefinition['modifiers']) => ({
    ctrl: Boolean(modifiers?.ctrl || modifiers?.meta),
    shift: Boolean(modifiers?.shift),
    alt: Boolean(modifiers?.alt),
  });

  const checkModifiers = useCallback((
    event: KeyboardEvent,
    requiredModifiers?: ShortcutDefinition['modifiers']
  ): boolean => {
    const pressed = getModifierState(event);
    const needed = getRequiredModifiers(requiredModifiers);

    // All required modifiers must be pressed
    if (needed.ctrl && !pressed.ctrl) return false;
    if (needed.shift && !pressed.shift) return false;
    if (needed.alt && !pressed.alt) return false;

    // No extra modifiers should be pressed
    if (!needed.ctrl && pressed.ctrl) return false;
    if (!needed.shift && pressed.shift) return false;
    if (!needed.alt && pressed.alt) return false;

    return true;
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (target.matches('input, textarea, [contenteditable="true"]')) {
        return;
      }

      for (const [, shortcut] of shortcuts) {
        if (shortcut.enabled === false) continue;

        const key = event.key.toLowerCase();
        const shortcutKey = shortcut.key.toLowerCase();

        if (key !== shortcutKey) continue;

        if (checkModifiers(event, shortcut.modifiers)) {
          event.preventDefault();
          shortcut.handlerRef.current();
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, checkModifiers]);

  const value = useMemo(
    () => ({
      shortcuts: Array.from(shortcuts.values()),
      registerShortcut,
      unregisterShortcut,
    }),
    [shortcuts, registerShortcut, unregisterShortcut]
  );

  return (
    <KeyboardShortcutContext.Provider value={value}>
      {children}
    </KeyboardShortcutContext.Provider>
  );
}

/**
 * Hook to register a keyboard shortcut.
 *
 * IMPORTANT: This hook is designed to handle unstable definitions gracefully.
 * The handler is stored in a ref, so even if you pass an inline function
 * like `handler: () => doSomething()`, the shortcut will NOT re-register
 * on every render. The latest handler will always be called.
 *
 * This means you don't need to memoize the definition object or wrap
 * the handler in useCallback - the hook handles this internally.
 *
 * NOTE: The shortcut is registered ONCE on mount and unregistered on unmount.
 * Context changes do NOT trigger re-registration (to break circular dependency).
 */
// eslint-disable-next-line react-refresh/only-export-components
export function useKeyboardShortcut(definition: ShortcutDefinition) {
  const context = useContext(KeyboardShortcutContext);

  if (!context) {
    throw new Error('useKeyboardShortcut must be used within KeyboardShortcutProvider');
  }

  // Store the handler in a ref to avoid re-registration when handler changes
  const handlerRef = useRef(definition.handler);

  // Keep the ref updated with the latest handler (no re-registration needed)
  handlerRef.current = definition.handler;

  // Store context functions in refs to avoid dependency on context object
  // This breaks the circular dependency: register → Map changes → context changes → re-register
  const registerRef = useRef(context.registerShortcut);
  const unregisterRef = useRef(context.unregisterShortcut);
  registerRef.current = context.registerShortcut;
  unregisterRef.current = context.unregisterShortcut;

  // Track registration state to ensure single registration
  const idRef = useRef<string | null>(null);

  useEffect(() => {
    // Create a stable definition using the handlerRef
    const stableDefinition: StableShortcutDefinition = {
      key: definition.key,
      modifiers: definition.modifiers,
      category: definition.category,
      description: definition.description,
      scope: definition.scope,
      enabled: definition.enabled,
      handlerRef,
    };

    // Register the shortcut once on mount
    idRef.current = registerRef.current(stableDefinition);

    return () => {
      // Unregister on unmount
      if (idRef.current) {
        unregisterRef.current(idRef.current);
        idRef.current = null;
      }
    };
    // Empty deps: register only on mount, unregister only on unmount
    // This breaks the circular dependency where context changes would trigger re-registration
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

// eslint-disable-next-line react-refresh/only-export-components
export function useShortcuts() {
  const context = useContext(KeyboardShortcutContext);
  if (!context) return [];
  return context.shortcuts;
}
