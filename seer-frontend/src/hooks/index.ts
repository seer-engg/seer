/**
 * Hooks Index
 *
 * Exports all custom hooks for the application.
 */

// Phase 2: Wrapper hooks removed, types moved to stores
// Integration tools - only export helper hook with business logic
export { useToolIntegration } from './useToolIntegration';

// Re-export types from stores for convenience
export type { ToolMetadata, ToolIntegrationStatus } from '@/stores/toolsStore';
export type { WorkflowListItem, WorkflowModel } from '@/stores/workflowStore';


// UI hooks
export { useToast } from './utility/use-toast';
export { useIsMobile } from './utility/use-mobile';
export { useMediaQuery } from './utility/useMediaQuery';
export { useQueryState } from './utility/useQueryState';
export {useConnectionValidation} from './useConnectionValidation';
export {useCanvasDragDrop} from './useCanvasDragDrop';

// Browser streaming hooks
export { useBrowserStream } from './useBrowserStream';
export { useBrowserProfiles } from './useBrowserProfiles';
export { useBrowserRecordings, useBrowserRecording, useBrowserRecordingEvents } from './useBrowserRecordings';
export { useBrowserReplay } from './useBrowserReplay';

// Trigger management hooks
export { useTriggerSubscriptions } from './useTriggerSubscriptions';
export { useTriggerToggle } from './useTriggerToggle';
