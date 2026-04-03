import { create } from 'zustand';
import type { StateCreator, StoreMutatorIdentifier } from 'zustand';

/**
 * Shared helper for creating Zustand stores with strong typing.
 * Works with additional middleware (persist, devtools, etc.).
 */
export function createStore<
  T,
  Mos extends [StoreMutatorIdentifier, unknown][] = [],
  Mcs extends [StoreMutatorIdentifier, unknown][] = []
>(initializer: StateCreator<T, Mos, Mcs>) {
  return create<T>()(initializer);
}


