import { humanId } from 'human-id';

/**
 * Generates a human-readable workflow name like "cosmic-butterfly" or "gentle-phoenix".
 * Uses adjective-noun-verb format for memorable, professional names.
 */
export function generateWorkflowName(): string {
  return humanId({
    separator: '-',
    capitalize: false,
  });
}
