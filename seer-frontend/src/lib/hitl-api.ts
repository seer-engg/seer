/**
 * HITL (Human-in-the-Loop) API Functions
 *
 * Provides API calls for fetching interrupt data and resuming interrupted workflows.
 */
import { backendApiClient } from './api-client';
import type { HitlInterruptData, HitlResumePayload } from '@/components/workflows/executions/types';

/**
 * Response from GET /runs/{run_id}/interrupt
 * Note: The API returns a flat structure (not nested under interrupt_data)
 */
export type GetInterruptResponse = HitlInterruptData;

/**
 * Response from POST /runs/{run_id}/resume
 */
export interface ResumeRunResponse {
  status: string;
  run_id: string;
}

/**
 * Fetches the pending interrupt data for a run that is in 'interrupted' status.
 *
 * @param runId - The ID of the interrupted run
 * @returns The interrupt data including display items and input fields
 */
export async function getRunInterrupt(runId: string): Promise<GetInterruptResponse> {
  return backendApiClient.request<GetInterruptResponse>(
    `/api/v1/runs/${runId}/interrupt`
  );
}

/**
 * Resumes an interrupted run with the user's responses.
 *
 * @param runId - The ID of the run to resume
 * @param payload - The user's responses to the input fields
 * @returns The updated run status
 */
export async function resumeRun(
  runId: string,
  payload: HitlResumePayload
): Promise<ResumeRunResponse> {
  return backendApiClient.request<ResumeRunResponse>(
    `/api/v1/runs/${runId}/resume`,
    { method: 'POST', body: payload as unknown as Record<string, unknown> }
  );
}
