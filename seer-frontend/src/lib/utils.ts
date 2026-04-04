import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function detectProvider(modelId: string): 'openai' | 'anthropic' {
  if (modelId.startsWith('claude-')) return 'anthropic';
  return 'openai'; // Default for gpt-, moonshotai/, etc.
}

/**
 * Formats a future datetime as a relative countdown string.
 * Returns "in Xh Ym" for hours or "in Xd Yh" for days.
 *
 * @param resetAt - ISO date string or null
 * @returns Formatted countdown string like "in 4h 30m" or "in 5d 12h", or null if invalid
 */
export function formatRelativeCountdown(resetAt: string | null | undefined): string | null {
  if (!resetAt) return null;

  const resetDate = new Date(resetAt);
  const now = new Date();
  const diffMs = resetDate.getTime() - now.getTime();

  if (diffMs <= 0) return "now";

  const totalMinutes = Math.floor(diffMs / (1000 * 60));
  const totalHours = Math.floor(totalMinutes / 60);
  const totalDays = Math.floor(totalHours / 24);

  if (totalDays >= 1) {
    const remainingHours = totalHours % 24;
    return `in ${totalDays}d ${remainingHours}h`;
  }

  const remainingMinutes = totalMinutes % 60;
  return `in ${totalHours}h ${remainingMinutes}m`;
}
