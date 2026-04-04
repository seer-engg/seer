export const formatGroupLabel = (value: string) => {
  const normalized = value.replace(/[_-]+/g, ' ').trim();
  if (!normalized) return 'Other';
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
};

export const normalizeIntegrationTypeKey = (value: string) => value.toLowerCase().trim();

export const getIntegrationTypeLabel = (integrationType: string) => {
  const key = normalizeIntegrationTypeKey(integrationType);
  switch (key) {
    case 'gmail':
      return 'Gmail';
    case 'googledrive':
    case 'google_drive':
      return 'Google Drive';
    case 'googlesheets':
    case 'google_sheets':
      return 'Google Sheets';
    case 'github':
    case 'pull_request':
      return 'GitHub';
    case 'supabase':
      return 'Supabase';
    default:
      return formatGroupLabel(integrationType);
  }
};

export const filterSystemPrompt = (content: string): string => {
  const systemPromptPattern = /I help you build, inspect, edit.*?Concretely I can:/s;
  let filtered = content.replace(systemPromptPattern, '').trim();
  filtered = filtered.replace(/What I can do \(high level\).*?$/s, '').trim();
  filtered = filtered.replace(/Your capabilities:.*?$/s, '').trim();
  return filtered;
};

const JSON_CODE_BLOCK_RE = /```json[\s\S]*?```/gi;
const ANY_CODE_BLOCK_RE = /```[\s\S]*?```/g;
const GENERIC_PROPOSAL_MESSAGE = 'Workflow proposal ready. Preview it in the canvas.';
const GENERIC_PROCESSING_MESSAGE = 'Processed your request.';

function summarizeThinkingSteps(thinking?: string[]): string | null {
  if (!thinking?.length) return null;
  const toolNames = new Set<string>();
  for (const step of thinking) {
    const match = step.match(/Calling tool '([^']+)'/);
    if (match) {
      toolNames.add(match[1].replace(/_/g, ' '));
    }
  }
  if (toolNames.size === 0) return null;
  const names = Array.from(toolNames).slice(0, 3);
  const suffix = toolNames.size > 3 ? ` and ${toolNames.size - 3} more` : '';
  return `Used ${names.join(', ')}${suffix}.`;
}

const stripCodeBlocks = (content: string, pattern: RegExp) => {
  return content.replace(pattern, '').trim();
};

const isLikelyJsonPayload = (content: string) => {
  const trimmed = content.trim();
  if (!trimmed) return false;
  const startsWithBrace = trimmed.startsWith('{') && trimmed.endsWith('}');
  const startsWithBracket = trimmed.startsWith('[') && trimmed.endsWith(']');
  if (!startsWithBrace && !startsWithBracket) {
    return false;
  }
  try {
    JSON.parse(trimmed);
    return true;
  } catch {
    return false;
  }
};

export const getDisplayableAssistantMessage = (content: string, fallback?: string, thinking?: string[]) => {
  if (!content) {
    if (fallback?.trim()) return fallback.trim();
    const thinkingSummary = summarizeThinkingSteps(thinking);
    if (thinkingSummary) return thinkingSummary;
    return GENERIC_PROCESSING_MESSAGE;
  }
  const withoutJsonBlocks = stripCodeBlocks(content, JSON_CODE_BLOCK_RE);
  if (withoutJsonBlocks) {
    return withoutJsonBlocks;
  }
  const withoutCodeBlocks = stripCodeBlocks(content, ANY_CODE_BLOCK_RE);
  if (withoutCodeBlocks) {
    return withoutCodeBlocks;
  }
  if (isLikelyJsonPayload(content)) {
    return fallback?.trim() || GENERIC_PROPOSAL_MESSAGE;
  }
  return fallback?.trim() || GENERIC_PROPOSAL_MESSAGE;
};
