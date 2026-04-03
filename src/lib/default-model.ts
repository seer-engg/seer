/**
 * Module-level store for the user's preferred default model.
 * Set by useAvailableModels when settings load; read by workflow utilities.
 */

export const FALLBACK_MODEL = 'qwen/qwen3-235b-a22b-2507';

let _userDefaultModel: string = FALLBACK_MODEL;

export function setUserDefaultModel(modelId: string) {
  _userDefaultModel = modelId;
}

export function getUserDefaultModel(): string {
  return _userDefaultModel;
}
