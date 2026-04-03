import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Mail } from 'lucide-react';

import { useTriggerIntegration } from '@/hooks/useTriggerIntegration';
import { GmailConfigState } from '../../../triggers/utils';
import { TriggerAccountSelector } from '../../../block-config/widgets/TriggerAccountSelector';

export const GmailDetailsSection: React.FC<{ triggerKey: string; triggerId: string }> = ({ triggerKey, triggerId }) => {
  const gmailIntegration = useTriggerIntegration(triggerKey, triggerId);

  return (
    <div className="rounded-md border border-dashed p-3 space-y-3 bg-muted/40">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Mail className="h-4 w-4" />
        Gmail connection
      </div>
      <TriggerAccountSelector
        triggerKey={gmailIntegration?.triggerKey ?? triggerKey}
        selectedConnectionId={gmailIntegration?.connectionId}
        onSelect={gmailIntegration?.onSelectAccount}
        onConnectAccount={gmailIntegration?.onConnect}
      />
    </div>
  );
};

export interface GmailConfigFormProps {
  gmailConfig: GmailConfigState;
  setGmailConfig: React.Dispatch<React.SetStateAction<GmailConfigState>>;
}

export const GmailConfigForm: React.FC<GmailConfigFormProps> = ({ gmailConfig, setGmailConfig }) => {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <div className="space-y-2">
        <Label className="text-xs uppercase text-muted-foreground">Label filters</Label>
        <Input
          value={gmailConfig.labelIds}
          placeholder="INBOX, UNREAD"
          onChange={(event) =>
            setGmailConfig((prev) => ({
              ...prev,
              labelIds: event.target.value,
            }))
          }
        />
        <p className="text-xs text-muted-foreground">Comma-separated Gmail label IDs (defaults to INBOX).</p>
      </div>
      <div className="space-y-2">
        <Label className="text-xs uppercase text-muted-foreground">Search query</Label>
        <Input
          value={gmailConfig.query}
          placeholder="from:vip@example.com is:unread"
          onChange={(event) =>
            setGmailConfig((prev) => ({
              ...prev,
              query: event.target.value,
            }))
          }
        />
        <p className="text-xs text-muted-foreground">Optional Gmail query appended to the poll watermark.</p>
      </div>
      <div className="space-y-2">
        <Label className="text-xs uppercase text-muted-foreground">Max results per poll</Label>
        <Input
          type="number"
          min={1}
          max={25}
          value={gmailConfig.maxResults}
          onChange={(event) =>
            setGmailConfig((prev) => ({
              ...prev,
              maxResults: event.target.value,
            }))
          }
        />
        <p className="text-xs text-muted-foreground">Between 1 and 25 messages per cycle.</p>
      </div>
      <div className="space-y-2">
        <Label className="text-xs uppercase text-muted-foreground">Overlap window (ms)</Label>
        <Input
          type="number"
          min={0}
          max={900000}
          step={1000}
          value={gmailConfig.overlapMs}
          onChange={(event) =>
            setGmailConfig((prev) => ({
              ...prev,
              overlapMs: event.target.value,
            }))
          }
        />
        <p className="text-xs text-muted-foreground">Re-read recent emails for dedupe protection.</p>
      </div>
    </div>
  );
};
