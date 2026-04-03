/**
 * Block selection payload types for workflow canvas
 */

/**
 * Base payload for block selection from palette or drag-and-drop
 */
export interface BlockSelectionPayload {
  type: string;
  label: string;
  config?: Record<string, unknown>;
}

/**
 * Trigger block payload with required triggerKey
 */
export interface TriggerBlockPayload extends BlockSelectionPayload {
  type: 'trigger';
  config: {
    triggerKey: string;
    [key: string]: unknown;
  };
}

/**
 * Type guard to check if payload is a trigger block
 */
export function isTriggerBlockPayload(
  payload: BlockSelectionPayload
): payload is TriggerBlockPayload {
  return (
    payload.type === 'trigger' &&
    payload.config !== undefined &&
    'triggerKey' in payload.config
  );
}
