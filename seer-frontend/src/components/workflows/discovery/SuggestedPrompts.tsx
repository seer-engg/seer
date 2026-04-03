import { DynamicIntegrationIcon } from '@/components/ui/DynamicIntegrationIcon';

interface SuggestedPromptsProps {
  onSelectPrompt: (prompt: string) => void;
}

const QUICK_START_CHIPS = [
  {
    label: 'Lead outreach agent',
    integrations: ['gmail', 'linkedin', 'google_sheets'],
    prompt:
      'Build an outreach agent that finds new leads in a Google Sheet, enriches them with LinkedIn data, and sends personalized cold emails via Gmail.',
  },
  {
    label: 'Email responder',
    integrations: ['gmail'],
    prompt:
      'Create an automated email responder that reads incoming Gmail messages, looks up answers in my knowledge base, and drafts helpful replies.',
  },
  {
    label: 'Meeting follow-up',
    integrations: ['google_calendar', 'gmail', 'google_docs'],
    prompt:
      'Build a workflow that watches my Google Calendar for completed meetings, generates a summary in Google Docs, and emails follow-up action items to attendees.',
  },
  {
    label: 'Content writer',
    integrations: ['linkedin'],
    prompt:
      'Create a content pipeline that researches trending topics on the web, writes LinkedIn posts on a schedule, and queues them for publishing.',
  },
  {
    label: 'CRM updater',
    integrations: ['notion', 'gmail'],
    prompt:
      'Build a workflow that captures form submissions, creates or updates entries in a Notion CRM database, and sends a confirmation email via Gmail.',
  },
  {
    label: 'Slack alert bot',
    integrations: ['gmail', 'slack'],
    prompt:
      'Create a bot that monitors a Gmail inbox for important emails matching certain criteria and sends instant alerts to a Slack channel.',
  },
];

export function SuggestedPrompts({ onSelectPrompt }: SuggestedPromptsProps) {
  return (
    <div className="flex flex-wrap justify-center gap-2 mb-2 max-w-xl mx-auto">
      {QUICK_START_CHIPS.map((chip) => (
        <button
          key={chip.label}
          type="button"
          onClick={() => onSelectPrompt(chip.prompt)}
          className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-3 py-1.5 text-left transition-all hover:shadow-sm hover:border-[hsl(var(--seer)/0.3)]"
        >
          <span className="text-xs font-medium whitespace-nowrap">{chip.label}</span>
          <div className="flex items-center gap-1">
            {chip.integrations.map((type) => (
              <DynamicIntegrationIcon
                key={type}
                integrationType={type}
                className="w-3.5 h-3.5"
              />
            ))}
          </div>
        </button>
      ))}
    </div>
  );
}
